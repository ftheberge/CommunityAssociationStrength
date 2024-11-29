import scipy.sparse as sparse 
from scipy.sparse import csr_matrix, lil_array
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

## from sparse membership matrix to dictionary of communities
## required format for the omega index function
def sparse2dict(M):
    dct = dict()
    T = M.transpose().tocsr()
    for i in range(len(T.indptr)-1):
        dct[i] = list(T.indices[T.indptr[i]:T.indptr[i+1]])
    return dct

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
    S = DegA.copy()
    S.data = np.array([binom.cdf(k=DegA.data[i]-1, n=Degrees[Nodes[i]], p=pA[DegA.indices[i]]) for i in range(len(Nodes))])    
    ## Internal edge fraction
    IEF = DegInv*DegA
    return IEF, Beta, C, S, DegA

## compute new community membership matrix given scores and threshold
## also apply condition w.r.t. minimum community degree
def score_to_memberships(S, DegA, threshold, min_deg_in=2):
    '''
    Input
    -----
    S: sparse (csr) score memberships to k comunities matrix (n by k)
    DegA: sparse (csr) matrix with degree of each node of each original community (n by k) 
    threshold: minimum "pass" score
    min_deg_in: minimum community degree for membership
    
    Output
    ------
    M: sparse (csr) community membership to k comunities matrix (n by k)
    '''    
    M = (1*(DegA>=min_deg_in)).multiply(1*(S>=threshold))
    return M



def incremental_conductance(A,new_set, old_set = None,old_numerator = None, old_denominator=None, eps = (0.01)**8):
    '''
    Input
    -----
    A: sparse (csr) adjacency matrix (n by n)
    DegA: sparse (csr) matrix with degree of each node of each original community (n by k) 
    new_set: (n by 1) sparse (csr) indicator function for membership in the "new" community to be evaluated.
    old_set: (n by 1) sparse (csr)  indicator function for membership in the "old" community that was already evaluated. Optional.
    old_numerator: float. Numerator in the conductance score for the "old" community (see below). Optional.
    old_denominator: float. Denominator in the conductance score for the "old" community (see below). Optional.
    
    Output
    ------
    numerator, denominator: floats. Numerator, denominator in the conductance score for the "new" community.

    Description
    -----

    Computes the conductance of the set new_set with respect to the adjacency matrix A. Recall that, after normalization, the usual conductance of a set S can be written as:
    Phi(S) = (\sum_{x \in S, y \notin S} A(x,y))/(\sum_{x \in S, y \in V} A(x,y)) = (1_{S} A 1_{S^{c}})/1_{S} A 1_{V}
    where 1_{FOO} is the indicator function for the set FOO. The numerator and denominator of this formula can be quickly updated for small changes in the set S. In particular, if S grows to a set T, we get:
    Numerator(T) - Numerator(S) = 1_{T} A(1_{T^{c}} - 1_{S^{c}}) + (1_{T} - 1_{S}) A 1_{S^{c}}
    '''
    if old_set is None:
        new_set_comp = 1*(new_set == 0)
        numerator = (new_set.transpose() @ A @ new_set_comp).toarray()[0][0]
        denominator = (new_set.transpose() @ A).sum()
        return numerator/(eps + denominator), (numerator, denominator)
    else:
        Delta_set = new_set - old_set
        new_set_comp = 1*(new_set == 0)
        Delta_N = (new_set.transpose() @ A @ (-Delta_set) + (Delta_set.transpose()) @ A @ new_set_comp).toarray()[0][0]
        Delta_D = (Delta_set.transpose() @ A).sum()
        numerator = old_numerator + Delta_N
        denominator = old_denominator + Delta_D
        return numerator/(eps + denominator), (numerator, denominator)
    return -1, -1

    
# Following class due to Jordan Barrett. Used to find connected components.
class UnionFind:
    def __init__(
        self,
        V,
    ):
        self.V = V
        self.parents = {v: v for v in self.V}
        self.sizes = {v: 1 for v in self.V}
 
    # Adds vertices and updates dictionaries.
    def add_vertices(self, V_added):
        self.V = self.V.union(V_added)
        # As it's written, if V_added contains vertices already in self.V then values will be overridden.
        for v in V_added:
            self.parents[v] = v
            self.sizes[v] = 1
        return
 
    # Returns the root of a vertex.
    def get_root(self, v):
        if self.parents[v] == v:
            return v
        else:
            return self.get_root(self.parents[v])
 
    # Sets root[v] the same for all v in S.
    def merge(self, S):  
        S = {self.get_root(v) for v in S}
        if len(S) > 1:
            v_max = list(S)[0]
            for v in S:
                if self.sizes[v] > self.sizes[v_max]:
                    v_max = v
            for v in S:
                self.parents[v] = v_max
            self.sizes[v_max] = sum([self.sizes[v] for v in S])
        return
 
    # Determines if the data structure is connected
    @property
    def is_connected(self):
        return max(self.sizes.values()) == len(self.V)
 
    # Returns the number of components
    @property
    def num_components(self):
        return len({self.get_root(v) for v in self.V})

def global_conn(A, new_set, old_set = None, old_UF = None):
    '''
    Input
    -----
    A: sparse (csr) adjacency matrix (n by n)
    new_set: (n by 1) sparse (csr) indicator function for membership in the "new" community to be evaluated.
    old_set: (n by 1) sparse (csr)  indicator function for membership in the "old" community that was already evaluated. Optional.
    old_UF: Object of type UnionFind. Datastructure that represents the connectedness of the set "old_set" if it exists. Optional.
    
    Output
    ------
    connected: Bool. True if "new_set" is connected with respect to the adjacency matrix A.
    new_UF: Object of type UnionFind. Datastructure that represents the connectedness of the set "new_set".
    '''
    if old_set is None:
        n = A.shape[0]
        UF = UnionFind(set([x for x in range(n)]))
        new_members = new_set.nonzero()[0]
        for member in new_members:
            S = set(A[member,:].nonzero()[1])
            UF.merge(S)
        return UF.is_connected, UF
    else:
        new_members = (new_set - old_set).nonzero()[0]
        for member in new_members:
            S = set(A[member,:].nonzero()[1])
            old_UF.merge(S)
        return old_UF.is_connected, old_UF
    return -1, -1

def column_rising_tide(A, s, global_incremental_scorer = incremental_conductance, global_req = global_conn, size_bias = None, enforce_req = False, min_deg_in = 2):
    # Order the non-zero indices of s according to their values. 
    # The parameter "size_bias" should be a function that takes in a (float: score, int: community-size) pair and returns a single (float: adjusted score).
    s_inds = s.nonzero()[0]
    s_vals = np.array([x[0] for x in s[s_inds,0].toarray()])
    sigma = np.argsort(-s_vals)
    sigma_inv = [x for x in range(len(sigma))]
    for i in range(len(sigma)):
        sigma_inv[sigma[i]] = i
    
    ordered_inds = np.array([s_inds[x] for x in sigma])

    # Compute the scores and requirements, in order.
    ordered_scores = []
    curr_set = lil_array(s.shape)
    curr_score_aux = None
    curr_req_aux = None
    for i in range(len(sigma)):
        new_set = curr_set
        new_set[ordered_inds[i],0] = 1
        if i == 0:
            curr_score, curr_score_aux = global_incremental_scorer(A, new_set.tocsr())
            if enforce_req:
                curr_req, curr_req_aux = global_req(A, new_set.tocsr())
            else:
                curr_req = True
        else:
            curr_score, curr_score_aux = global_incremental_scorer(A, new_set.tocsr(),old_set = curr_set.tocsr(), old_numerator = curr_score_aux[0],  old_denominator = curr_score_aux[1]) # TODO: This is a very restrictive way of passing extra arguments. Fix if/when we care.
            if enforce_req:
                curr_req, curr_req_aux = global_req(A, new_set.tocsr(),old_set = curr_set.tocsr(), old_UF = curr_req_aux)
            else:
                curr_req = True
        if curr_req:
            ordered_scores.append(curr_score)
        else:
            ordered_scores.append(-(i+1))
        curr_set = new_set

    if size_bias is not None:
        ordered_scores = [size_bias(ordered_scores[i],i) for i in range(len(ordered_scores))]

    # Find the index with the best score, subject to the requirement curr_req
    ordered_scores = np.array(ordered_scores)
    best_ind = np.argsort(-ordered_scores)[0]
    best_set_inds = [s_inds[sigma_inv[j]] for j in range(best_ind)]
    best_set_vec = lil_array(s.shape)
    for i in best_set_inds:
        best_set_vec[i,0] = 1.0
    return best_set_vec
        
## compute new community membership matrix given vertex-community scores, using a "global" score function to estimate per-community thresholds.
## also apply condition w.r.t. minimum community degree.
## with default values, should function as a plug-in replacement to "score_to_memberships". WARNING: due to shape of input, that probably doesn't work right now, but presumably this is easy to fix.
def rising_tide(A, S, global_incremental_scorer = incremental_conductance, global_req = global_conn, size_bias = None, enforce_req = False, min_deg_in = 2):
    '''
    Input
    -----
    A: sparse (csr) adjacency matrix (n by n)
    S: sparse (csr) score memberships to k comunities matrix (n by k)
    global_incremental_scorer: a function that should take in a tuple of the form (A, M[:,i]) and optional extra data, then return a score and an auxiliary tuple.
    global_req: a function that should take in a tuple of the form (A,M[:,i]) and optional extra data, then return a true/false value and an auxiliary tuple.
    size_bias: a function that should take in a number and return another number. This is added as a bias towards larger/smaller communities. Optional.
    min_deg_in: minimum community degree for membership
    
    Output
    ------
    M: sparse (csr) community membership to k comunities matrix (n by k)
    '''    
    res = lil_array(S.shape)
    for i in range(S.shape[1]):
        res[:,i] = column_rising_tide(A, S[:,i], global_incremental_scorer = incremental_conductance, global_req = global_conn, size_bias = size_bias, enforce_req = enforce_req, min_deg_in = 2).toarray()
    res = csr_matrix(res, dtype=int)
    return res
