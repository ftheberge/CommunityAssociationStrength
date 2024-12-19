import numpy as np
from scipy.sparse import lil_array, csr_matrix
import pandas as pd
import pickle
import igraph as ig
from collections import Counter
from matplotlib import pyplot as plt
import copy
from pathlib import Path
import glob
import utils
from CAS import *

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

import random
import csv
import subprocess
import sys
import os
# Workflow:
# 1. Build features for each vertex/community pair (e.g. "build_basic_features" function).
# 2. When training, go from ground-truth community label y to the appropriate explosion (there are two options here: 1. literally expanding the vector y, or 2. expanding the vector y based on communities estimated from Louvain )
# 3. Given features and a model, train.
# 4. Go back from exploded labels to a community matrix.

#######################
# 0. Helper Functions #
#######################


#path = '../Datasets/ABCDoo/'
def readGraph(xi=0.5, eta=1.5, rep=1, path = '/data/ABCDoo/'):
    ## read edges, build graph
    fn = path+'networkfile'+str(xi)+'_'+str(eta)+'_'+str(rep)+'.txt'
    Edges = pd.read_csv(fn, sep='\t', header=None)-1
    G = ig.Graph.DataFrame(Edges, directed=False)
    ## read communities
    fn = path+'communityfile'+str(xi)+'_'+str(eta)+'_'+str(rep)+'.txt'
    L = []
    with open(fn, "r") as infile:
        for line in infile:
            x = line.split('\t')
            L.append([int(y)-1 for y in x[1].rstrip()[1:-1].split(',')]) ## map to 0-based
    G.vs['comms'] = L
    G.vs['n_comms'] = [len(x) for x in G.vs['comms']]
    return G

## given list of node memberships, return list of communities
def mems2comms(X):
    nc = int(max(set([i for j in X for i in j]))+1)
    n = len(X)
    L = [[] for _ in range(nc)]
    for i in range(n):
        for j in X[i]:
            L[int(j)].append(i)
    return L

## given matrix of node membership, return list of node memberships
def m2mems(M, thresh =0.5):
    res = []
    for i in range(M.shape[0]):
        temp = []
        for j in range(M.shape[1]):
            if M[i,j] > thresh:
                temp.append(j)
        res.append(temp)
    return res

##############################################
# 1. Build (vertex,community) pair features. #
##############################################

def build_basic_features(A, Scores, Global_Features, y_mat = None, include_rank = True, include_context = True):
    """
    Note: column features are in the order: (A) Loop through Scores, including (i) raw score, (ii) biggest other score, (iii) rank in that order. (B) Loop through global features.
    """
    k = Scores[0].shape[1] # Number of communities
    n = A.shape[0] # Number of vertices
    res = None 
    res_y = None

    # Do the simple features
    for j in range(k):
        builder = np.zeros((n,2))  # TBD: Kill the first two columns before appending.
        if include_context:
            mask = lil_array((n,k))
            mask[:,j] = 1.0
        for Score_Mat in Scores:
            builder = np.hstack((builder,Score_Mat[:,j].toarray()))
            if include_context:
                temp = Score_Mat - Score_Mat.multiply(mask) # Mask the current column
                builder = np.hstack((builder, temp.max(axis=1).toarray()))
            if include_rank:
                thresh = Score_Mat[:,j].toarray()
                temp = 1.0*(Score_Mat.toarray() > thresh)
                temp2 = temp.sum(axis=1).reshape((temp.shape[0],1))
                builder = np.hstack((builder, temp2))
        for Global_Feature_Mat in Global_Features:
            builder = np.hstack((builder,Global_Feature_Mat[:,j].toarray()))
        builder = builder[:, 2:(builder.shape[1])]
        if res is None:
            res = np.zeros((2,builder.shape[1]))  # TBD: Kill the first two rows before returning.
        res = np.vstack((res,builder))
    if y_mat is not None:
        if type(y_mat) is not np.ndarray:
            y_mat = y_mat.toarray()
        
        res_y = None
        for i in range(k):
            if res_y is None:
                res_y = y_mat[:,i]
            else:
                res_y = np.hstack((res_y,y_mat[:,i])) # WARNING: looks transpose...

    res = res[2:(res.shape[0]),:]# Killing first two rows
    if y_mat is None:
        return res
    else:
        return res, res_y

def jordan_features(n,Community_Features):
    """
    WARNING: NOT USED and PROBABLY NOT EFFICIENT - really just a placeholder/reminder.
    n: integer, number of vertices
    Community_Features: k by m matrix, were k is the number of communities and m is the number of features of interest. Jordan suggested community size (and/or community fraction) as a possible feature.
    """
    res = np.zeros((n*k,m))
    for i in range(m):
        for j in range(k):
            res[(j*n):((j+1)*n),i] = Community_Features[j,i]
    return res

#######################
# 2. Label explosion. #
#######################

# First, the "default" method for a simple binary matrix.

def explode_labels(M):
    """
    Given an (n by k) community-membership matrix, return a length-nk vector 
    """
    return M.reshape((M.shape[0]*M.shape[1],1),order="F")

# Next, a method used only for training: try to get the best-possible indicator for whether an estimated label M is "correct" given ground-truth labels y.
# I assume M is the output of some algorithm (e.g. Leiden) while y is a list of lists of ground-truth labels (i.e. y[x] is the list of communities that x belongs to).
# In our code, y = g.vs["comms"].
# In the following discussion, I imagine y is in the same format as M (so that y[x,i] indicates if vertex x is in community i)

# Note that the labels can disagree arbitrarily - M and y will generally not even have the same number of communities, so it isn't obvious whether a label should be "right."
# For this reason, the "first step" is to decide what "right" means.
# One natural choice: let's say we want to decide if vertex v belongs to community i (where "i" is a community according to M but not y).
# 1. Denote by S(i) = {j: M[i,j] = 1} the vertices in community i, according to M. (this dict as k_m entries)
# 2. Denote by N[j,:] = y[j,:]/sum(y[j,:]) the row-normalized version of y. (This is n by k_y).
# 3. Denote by P[i,:] = mean(N[S(i),:]) the average of the normalized community matrices. (this is k_m by k_y)
# 4. Denote the association score between vertex v and M-community j by the cosine similarity of N[v,:] and P[j,:]. If it is large, then say v belongs to community j.

# In the following, I use k_m, k_y to denote the number of communities according to M and y= g_comm.

def explode_training_labels(M,g_comm):
    # First, turn g_comm into a nice sparse matrix of shape (n by k_y). 
    k_y = max(g_comm)[0] + 1
    n = M.shape[0]
    k_m = M.shape[1]
    y = lil_array((n,k_y))
    for i in range(len(g_comm)):
        for j in g_comm[i]:
            y[i,j] = 1

    # Compute "S" as a dict with k_m keys.
    S = dict()
    for i in range(k_m):
        S[i] = []

    row, col = M.nonzero()
    for i in range(len(row)):
        S[col[i]].append(row[i])
    
    # Compute N, a matrix of shape (n by k_y)
    y = y.tocsr()
    y_means = y.mean(axis=1).reshape((n,1))
    N = y/y_means
    N = N.tocsr()

    # Compute P, a matrix of shape (k_m by k_y)
    # Do this by defining k_y by n matrix "mask" whose i'th row is 1_{S(i)}/|S(i)| and taking P = mask @ N.
    # ZZX: https://stackoverflow.com/questions/29629821/sum-over-rows-in-scipy-sparse-csr-matrix
    col = []
    row = []
    dat = []
    for i in range(k_m):
        temp = len(S[i])
        for j in S[i]:
            row.append(i)
            col.append(j)
            dat.append(1.0/temp)
    mask = csr_matrix((dat, (row, col)), shape=(k_m,n))
    P = mask @ N
    # I don't know how to make this part sparse... TBD.
    res = np.zeros((n,k_m))
    N_np = N.toarray()
    P_np = P.toarray()

    for v in range(n):
        nv = np.sqrt(N_np[v,:].dot(N_np[v,:]))
        for j in range(k_m):
            pj = np.sqrt(P_np[j,:].dot(P_np[j,:]))
            res[v,j] = N_np[v,:].dot(P_np[j,:])/(nv*pj)

    return res

#################
# 3. Inference. #
#################

# Fit: read the graph with rep = 1 to learn a classifier.

def fit_naive_threshold(xi=0.5, eta=1.5, n_estimators=100, learning_rate=1.0,max_depth=1, random_state=42,fudge=0.3):
    g = readGraph(xi=xi, eta=eta, rep=1)
    # Extract basic information in our desired format.
    temp = []
    g_comm = g.vs["comms"]
    for comms in g_comm:
        for comm in comms:
            temp.append(comm)
    
    all_labels = np.unique(temp)
    n_labels = len(all_labels)
    comm_bar = 1.0*len(temp)/len(g_comm)
    
    y = np.zeros((len(g_comm),n_labels))
    for i in range(len(g_comm)):
        for comm in g_comm[i]:
            y[i,comm] = 1

    A = g.get_adjacency_sparse()
    y_sparse = csr_matrix(y)
    ### Get initial partition and adjacency
    L = g.community_leiden(objective_function='modularity').membership
    M = partition2sparse(L)
    A = g.get_adjacency_sparse()
    
    ### Compute Beta and C scores; also get degrees w.r.t. initial partition
    IEF, Beta, C, Pv, DegPart = CAS(A, M, alpha=1)

    # Compute features
    features = build_basic_features(A, [Beta,IEF,C,Pv], [DegPart])
    # Compute proxy for ground truth, 
    M_fixed = explode_training_labels(M,g_comm)
    # Threshold so that each vertex is part of the right number of communities
    thresh = np.quantile(M_fixed,1 - 1.0*comm_bar/M.shape[1])*fudge
    print(thresh)
    M_thresh = 1.0*(M_fixed > thresh)
    labels = explode_labels(M_thresh)

    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,max_depth=max_depth, random_state=random_state).fit(features, labels) 
    return clf

def test_naive_threshold(classifier, xi=0.5,eta=1.5):
    res = []
    for i in range(2,10):
        g = readGraph(xi=xi, eta=eta, rep=i)
        # Extract basic information in our desired format.
        temp = []
        for comms in g.vs["comms"]:
            for comm in comms:
                temp.append(comm)
        
        all_labels = np.unique(temp)
        n_labels = len(all_labels)
        
        y = np.zeros((len(g.vs["comms"]),n_labels))
        for i in range(len(g.vs["comms"])):
            for comm in g.vs["comms"][i]:
                y[i,comm] = 1
    
        ### Get initial partition and adjacency
        L = g.community_leiden(objective_function='modularity').membership
        M = partition2sparse(L)
        A = g.get_adjacency_sparse()
        ### Compute Beta and C scores; also get degrees w.r.t. initial partition
        IEF, Beta, C, Pv, DegPart = CAS(A, M, alpha=1)
        features = build_basic_features(A, [Beta,IEF,C,Pv], [DegPart])
        
        # More silly reformatting
        k = IEF.shape[1] # Number of communities
        res_y = None
        for i in range(k):
            if res_y is None:
                res_y = y[:,i]
            else:
                res_y = np.hstack((res_y,y[:,i])) # WARNING: looks transpose...
        
        labels = res_y
        res.append(classifier.score(features, labels))
    return res
        

#########################
# 4. Label contraction. #
#########################

# Go from the length (k_m*n) vector y back to an (n by k_m) class matrix

def contract_labels(y,n):
    """
    inverts explode_labels
    TBD: Check that this is true!
    """
    return y.reshape((n,int(y.shape[0]/n)),order="F")

###############
# Experiments #
###############

# ZZX: do experiments that look at arbitrary functions of the 13 features, with arbitrary threshold. Anything "good"?
#