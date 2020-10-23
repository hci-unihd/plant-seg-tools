import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def find_neighborhood(e, edges):
    neighborhood = []
    for _e_a, _e_b in edges:
        if _e_a == e:
            neighborhood.append(_e_b)
            
        if _e_b == e:
            neighborhood.append(_e_a)
    return neighborhood


def get_is_surface(neighborhood):
    return True if 0 in neighborhood else False


def spath_to_node(e, edges, destination=0):    
    # this line works only for meshes!!! (for generic graph to adjacency this would not work)
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(edges.max()+1, edges.max()+1))
    A = A + A.T
    return shortest_path(A, indices=e)[destination]


def global_features_extractor(label, edges):
    if type(edges) is list:
        edges = np.array(edges)
        
    neighborhood = find_neighborhood(label, edges)
    neighborhood_size = len(neighborhood)
    is_surface = get_is_surface(neighborhood)
    spath_surface = spath_to_node(label, edges, destination=0)
    
    global_features = {'neighborhood': neighborhood,
                       'neighborhood_size': neighborhood_size,
                       'is_surface': is_surface,
                       'spath_surface': spath_surface}
    
    return global_features
