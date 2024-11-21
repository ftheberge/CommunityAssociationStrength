import scipy.sparse as sparse 
from scipy.sparse import csr_matrix
import numpy as np
from scipy.stats import binom

## top 2 for each row in an array
def top2arg(X):
    return np.array([(-x).toarray().flatten().argsort()[:2] for x in X])
def top2(X):
    return np.array([sorted(x.toarray().flatten(), reverse=True)[:2] for x in X])

## avoid values close to 0 or 1
def clip(v, epsilon=1e-7):
    v[v<epsilon] = epsilon
    v[v>(1-epsilon)] = 1-epsilon
    return v

## from partition (memberships) to csr sparse matrix (n by k)
def partition2sparse(partition):
    n = len(partition)
    X = sparse.csr_matrix( (np.repeat(1,n), (np.arange(n, dtype=int),partition)), 
                           shape=(n, max(partition)+1) )
    return X

## A: Adjacency matrix (n by n, csr)
## M: sparse matrix of community memberships (n by k, csr)
def CAS(A, M, alpha=1):
    '''
    Input
    -----
    A: sparse (csr) adjacency matrix (n by n)
    M: sparse (csr) initial community matrix, membership to k comunities (n by k); this is usually a partition
    alpha: [0,1], weight of the 1-hop neighbourhood (default=1)
    
    Output
    ------
    IEF: sparse (csr) internal edge fraction by (n by k), all values positive.
    Beta: sparse (csr) community score matrix (n by k), all values positive (normalized IEF).
    C: sparse (csr) community score matrix (n by k), all values positive (stdv given beta)
    S: sparse(csr) community score matrix (n by k), all values positive (normalized probabilities)
    DegA: sparse (csr) matrix with degree of each node of each original community (n by k) 
    '''
    ## compute all deg_A(v)
    Degrees = np.array(A.sum(axis=1)).flatten()
    if alpha==1:
        DegA = A*M
    else:
        DegA = ((1-alpha)*(A*A) + alpha*A)*M
    ## compute all 1/deg(v)
    DegInv = sparse.diags(1/Degrees).tocsr()
    ## compute all VolA and volV and save the ratios as pA
    VolA = Degrees*M
    VolV = A.sum()
    pA = clip(VolA/VolV, epsilon=1e-7)
    ## Compute Beta given the above
    Beta = DegInv*DegA
    Beta.data = Beta.data - pA[Beta.indices]
    Beta[Beta<0] = 0 ## to keep sparse matrix    
    ## compute all sqrt(deg(v))
    DegSqrt = sparse.diags(np.sqrt(Degrees)).tocsr()
    ## compute C
    p = 1/np.sqrt(pA*(1-pA))
    C = DegSqrt * Beta * sparse.diags(p).tocsr()
    ## p-values
    N = DegA.shape[0]
    Ptr = DegA.indptr
    Nodes = np.repeat(np.arange(N), np.diff(DegA.indptr))
    Pv = DegA.copy()
    Pv.data = np.array([binom.cdf(k=DegA.data[i]-1, n=Degrees[Nodes[i]], p=pA[DegA.indices[i]]) for i in range(len(Nodes))])    
    ## Internal edge fraction
    IEF = DegInv*DegA
    return IEF, Beta, C, Pv, DegA

## compute new community membership matrix given scores and threshold
## also apply condition w.r.t. minimum community degree
def score_to_memberships(S, DegA, threshold, min_deg_in=2):
    '''
    Input
    -----
    S: sparse (csr) score, memberships to k comunities matrix (n by k)
    DegA: sparse (csr) matrix with degree of each node of each original community (n by k) 
    threshold: minimum "pass" score
    min_deg_in: minimum community degree for membership
    
    Output
    ------
    M: sparse (csr) community membership matrix (n by k)
    '''    
    M = (1*(DegA>=min_deg_in)).multiply(1*(S>=threshold))
    return M
