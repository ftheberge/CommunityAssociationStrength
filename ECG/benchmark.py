import igraph as ig
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import roc_auc_score as AUC
from tqdm import tqdm

from itertools import product
import sys
import os
sys.path.append('../')
import CAS
from ecg_igraph import *

# ABCD+o only in Julia for now - update path below as needed

## local:
# abcd_path = '/Users/francois/ABCD/ABCDo/ABCDGraphGenerator.jl/utils/'
# julia = '/Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia '

## Dev02:
abcd_path = '/home/rdewolfe/research/ABCDGraphGenerator.jl/utils/'
julia = '/home/rdewolfe/.juliaup/bin/julia '

def _run_julia_abcd(n=1000, xi=0.3, delta=5, zeta=0.5, gamma=2.5, s=25, tau=0.825, beta=1.5, seed=123, nout=0):
    D = int(n**zeta)
    S = int(n**tau) 
    rdm = str(np.random.choice(100000))
    fn_deg = 'deg_'+rdm+'.dat'
    fn_cs = 'cs_'+rdm+'.dat'
    fn_comm = 'comm_'+rdm+'.dat'
    fn_net = 'net_'+rdm+'.dat'
    ## generate graph
    cmd = julia+abcd_path+'deg_sampler.jl '+fn_deg+' '+str(gamma)+' '+str(delta)+' '+str(D)+' '+str(n)+' 1000 '+str(seed)
    os.system(cmd+' >/dev/null 2>&1')
    cmd = julia+abcd_path+'com_sampler.jl '+fn_cs+' '+str(beta)+' '+str(s)+' '+str(S)+' '+str(n)+' 1000 '+str(seed)+' '+str(nout)
    os.system(cmd+' >/dev/null 2>&1')
    cmd = julia+abcd_path+'graph_sampler.jl '+fn_net+' '+fn_comm+' '+fn_deg+' '+fn_cs+' xi '+str(xi)+' false false '+str(seed)+' '+str(nout)
    os.system(cmd+' >/dev/null 2>&1')
    g = ig.Graph.Read_Ncol(fn_net,directed=False)
    c = np.loadtxt(fn_comm,dtype='uint16',usecols=(1))
    ## ground-truth communities
    gt = [c[int(i['name'])-1]-1 for i in g.vs]
    g.vs['gt'] = gt
    cmd = 'rm *_'+rdm+'.dat'
    os.system(cmd+' >/dev/null 2>&1')
    return g



#Actually running the benchmark

REP = 10
num_nodes = 10000
num_out = 0
delta = 5
min_comm = 100
alpha = 2
save_every = 5
save_file = "ecg_benchmarks.feather"

XIs = [.5, .525, .55, .575, .6, .625, .65, .675, .7]
pouts = [.01, .1, .3, .5]
data = []


df = None
in_checkpoint = False
try:
    df = pd.read_feather(save_file)
    print(f"Successfully loaded {len(df)} rows.")
    in_checkpoint = True
except FileNotFoundError:
    pass


n = len(XIs)*len(pouts)*REP
pbar = tqdm(total=n)
i = 0
for xi, pout, rep in product(XIs, pouts, range(REP)):
    n_out = int(pout*num_nodes)

    # Skip iteration if present in loaded dataframe
    if df is not None and in_checkpoint:
        tail = df.iloc[-1]
        if tail.xi != xi or tail.n_out != n_out or tail.rep != rep:
            pbar.update()
            continue
        else:
            in_checkpoint = False
            pbar.update()
            continue

    # Get stars from one graph
    g = _run_julia_abcd(n=num_nodes, xi=xi, delta=delta, gamma=2.5, beta=1.5, zeta = 0.5, s=min_comm, tau = 0.767, nout=n_out)
    is_outlier = np.array(g.vs["gt"]) == 0
    gt = np.array(g.vs["gt"])

    ief, beta, c, p, ecg = ensemble_cas_edge_weights(g)
    #ecg = np.array(g.community_ecg().W)
    options = [
        [ief, "IEF"],
        [beta, "BETA"],
        [c, "C"],
        [p, "P"],
        [ecg, "ECG"]
    ]
    for edge_weights, name in options:
        clustering = np.array(cluster_edges(g, edge_weights).membership)
        overall_outlier, community_outlier = outlier_scores(g, edge_weights, clustering)
        data.append([name, xi, n_out, rep, "Overall Outlier AUC", AUC(is_outlier, overall_outlier)])
        data.append([name, xi, n_out, rep, "Community Outlier AUC", AUC(is_outlier, community_outlier)])
        data.append([name, xi, n_out, rep, "Non-outlier AMI", AMI(gt[is_outlier == 0], clustering[is_outlier == 0])])

        # Test CAS for outlier detection on the CAS-ECG clustering
        this_ief, this_beta, this_c, this_p, degs = CAS.CAS(g.get_adjacency_sparse(), CAS.partition2sparse(clustering))
        this_options = [
            [this_ief, "IEF Outlier AUC"],
            [this_beta, "BETA Outlier AUC"],
            [this_c, "C Outlier AUC"],
            [this_p, "P Outlier AUC"]
        ]
        for cas, this_name in this_options:
            attatchment_to_community = [cas[i, clustering[i]] for i in range(len(clustering))]
            data.append([name, xi, n_out, rep, this_name, AUC(is_outlier, attatchment_to_community)])
    
    pbar.update()
    i += 1
    if i // save_every == 1:
        new = pd.DataFrame(data, columns=['name','xi','n_out', 'rep', 'task','score'])
        if df is not None:
            df = pd.concat([df, new])
        else:
            df = new
        df.to_feather(save_file)
        data = []


new = pd.DataFrame(data, columns=['name','xi','n_out', 'rep', 'task','score'])
if df is not None:
    df = pd.concat([df, new])
df.to_feather(save_file)
