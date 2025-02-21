import numpy as np
from numpy.typing import NDArray
from collections import defaultdict

# DBLP Params
n = 317080
nout = 56082

x_min = 1
x_max = 124
beta_1 = 2.1

c_min = 6
c_max = 7556
beta_2 = 2.28

output_file = "dblp_cbk.txt"

n = n-nout

print(f"n: {n}, x_min:{x_min}, x_max: {x_max}, beta_1: {beta_1}, c_min: {c_min}, c_max: {c_max}, beta_2: {beta_2}")

def powerlaw_distribution(choices: NDArray[np.float64], intensity: float) -> NDArray[np.float64]:
    # Helper function taken from abcd-graph
    dist: NDArray[np.float64] = (choices ** (-intensity)) / np.sum(choices ** (-intensity))
    return dist


def expected_value(events, probabilities):
    return np.sum(events * probabilities)


# Assign nodes a (non-zero) number of communities
print("Making Number of Communities.")
x_available = np.arange(x_min, x_max + 1, dtype=float)
x_probabilities = powerlaw_distribution(x_available, beta_1)
number_of_coms = np.random.choice(x_available, size=n, p=x_probabilities)
assert np.min(number_of_coms) >= x_min
assert np.max(number_of_coms) <= x_max

# Get community sizes
print("Making Community Sizes.")
x_0 = expected_value(x_available, x_probabilities)
m_available = np.arange(c_min, c_max+1, dtype=float)
m_probabilities = powerlaw_distribution(m_available, beta_2)
m_0 = expected_value(m_available, m_probabilities)
num_coms = int(n*x_0/m_0)
com_sizes = np.random.choice(m_available, size=num_coms, p=m_probabilities)
print(f"    Expected Coms per node: {x_0:.2f}, Expected Com sizs: {m_0:.2f}, Number of Coms: {num_coms}.")

# Adjust community sizes to satisfy bipartite degree constaint
required_adjustment = -int(np.sum(com_sizes) - np.sum(number_of_coms))
print(f"    Size of adjustment: {required_adjustment}.")
# Make the required adjustment by randomly selecting a community and changing it's size if allowed
max_tries = int(abs(required_adjustment)*100)
random_coms = np.random.choice(np.arange(num_coms), size=max_tries)
i = 0
while required_adjustment != 0:
    random_com = random_coms[i]
    #print(f"Round {i} has com {random_com} of size {com_sizes[random_com]}.")
    if required_adjustment < 0 and com_sizes[random_com] > c_min:
        com_sizes[random_com] -= 1
        required_adjustment += 1
    elif required_adjustment > 0 and com_sizes[random_com] < c_max:
        com_sizes[random_com] += 1
        required_adjustment -= 1
    
    if i == max_tries-1:
        break

    i += 1
#print(f"{i} fast rounds tried. {required_adjustment} adjustment remaining.")

# More expensive randomly selecting communities that satisfy the constraints
if required_adjustment != 0:
    while required_adjustment != 0:
        if required_adjustment < 0:
            com_options = np.arange(num_coms)[com_sizes > c_min]
            if len(com_options) == 0:
                raise ValueError("Community sizes cannot be matched to constaints. Try again.")
            random_com = np.random.choice(com_options, size=1)
            com_sizes[random_com] -= 1
            required_adjustment += 1
        else:
            com_options = np.arange(num_coms)[com_sizes < c_max]
            if len(com_options) == 0:
                raise ValueError("Community sizes cannot be matched to constaints. Try again.")
            random_com = np.random.choice(com_options, size=1)
            com_sizes[random_com] += 1
            required_adjustment -= 1

assert np.sum(com_sizes) == np.sum(number_of_coms)
assert np.min(com_sizes) >= c_min
assert np.max(com_sizes) <= c_max


# Build edges with the configuration model.
print("Building Edges.")
node = np.empty(int(np.sum(number_of_coms)), dtype="int64")
start = 0
for i, num in enumerate(number_of_coms):
    end = int(start+num)
    node[start:end] = i
    start += int(num)
np.random.shuffle(node)

com = np.empty(int(np.sum(com_sizes)), dtype="int64")
start = 0
for i, num in enumerate(com_sizes):
    end = int(start+num)
    com[start:end] = i
    start += int(num)
np.random.shuffle(com)

edges = np.vstack([node, com]).transpose()


# Check community size and number constraints.
n_coms = defaultdict(int)
com_sizes = defaultdict(int)
for i in range(edges.shape[0]):
    n_coms[edges[i, 0]] += 1
    com_sizes[edges[i, 1]] += 1
n_coms = np.array(list(n_coms.values()))
com_sizes = np.array(list(com_sizes.values()))
assert np.min(n_coms) >= x_min
assert np.max(n_coms) <= x_max
assert np.min(com_sizes) >= c_min
assert np.max(com_sizes) <= c_max


# Check for collisions (no loops due to bipartite nature). We take advatnage of the fact each edge
# is in the form (node, com) to hash.
print("Resolving Conflicts.")
def get_bad_edge_ids(edges):
    seen_edges = set()
    recycle_ids = set()
    for i in range(edges.shape[0]):
        edge = tuple(edges[i, :])
        if edge in seen_edges:
            recycle_ids.add(i)
        else:
            seen_edges.add(edge)
    return recycle_ids

max_recycle_tries = 1000
i = 0
recycle_ids = get_bad_edge_ids(edges)
print(f"    Recycle {len(recycle_ids)} of {edges.shape[0]} edges.")
while len(recycle_ids) > 0:
    for edge_id in recycle_ids:
        other_id = np.random.choice(np.arange(edges.shape[0]), size=1)[0]
        if edge_id == other_id:  # TODO fix random collision
            continue
        edges[edge_id, 1], edges[other_id, 1] = edges[other_id, 1], edges[edge_id, 1]
    recycle_ids = get_bad_edge_ids(edges)
    if i == max_recycle_tries:
        raise ValueError(f"Couldn't resolve collisions, {len(recycle_ids)}. Try again.")
    i += 1
print(f"    Resolved collisions in {i} rounds.")


# Check community size and number constraints.
n_coms = defaultdict(int)
com_sizes = defaultdict(int)
for i in range(edges.shape[0]):
    n_coms[edges[i, 0]] += 1
    com_sizes[edges[i, 1]] += 1
n_coms = np.array(list(n_coms.values()))
com_sizes = np.array(list(com_sizes.values()))
assert np.min(n_coms) >= x_min
assert np.max(n_coms) <= x_max
assert np.min(com_sizes) >= c_min
assert np.max(com_sizes) <= c_max


# Write results to a file
print("Writing Results.")
coms = [[] for _ in range(n)]
com_index_bump = 1
if nout > 0:
    com_index_bump = 2  # reserve community 1 for outliers
for i in range(edges.shape[0]):
    coms[edges[i, 0]].append(edges[i, 1] + com_index_bump) # Reindex coms to 1 for consistency with julia

with open(output_file, "w") as f:
    for i, c in enumerate(coms):
        f.write(f"{i+1}\t{c}\n")  # Reindex nodes to 1 for consistency with julia

    if nout > 0:
        next_id = len(coms)+1
        for i in range(nout):
            f.write(f"{i}\t[1]\n")
            i += 1

print("Success!")


