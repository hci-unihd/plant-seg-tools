import numpy as np

from skimage import measure
from sklearn.decomposition import PCA

from numba import jit
from scipy.special import sph_harm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from trimesh import curvature
import trimesh


def get_com(coord):
    """returns center of mass from coordinates"""
    return np.mean(coord, axis=0)


def get_mu(coord):
    """return moments from coordinates"""
    return measure.moments_coords(coord)


def get_inertia(mask, mu=None):
    """compute inertia tensor and eigenvalues from mask, if moments are give the function is much faster"""
    if mu is None:
        mu = measure.moments(mask)

    inertia_tensor = measure.inertia_tensor(mask, mu)
    inertia_eigen = measure.inertia_tensor_eigvals(mask, mu=mu, T=inertia_tensor)
    return inertia_tensor, inertia_eigen


def rotate_points(coord, rotation_matrix, center_of_rotation=0):
    """takes coord vector (N, 3) and apply a rotation to all pints"""
    return rotation_matrix.dot((coord - center_of_rotation).T).T + center_of_rotation


def get_pca_components(verts):
    """compute the pca component"""
    pca = PCA(n_components=3)
    pca.fit(verts)

    axis = pca.components_
    inv_axis = np.linalg.inv(axis)
    return axis, inv_axis


@jit()
def _get_triangle_area(x, y, z):
    """private method: compute triangle area given three points"""
    a = np.sqrt(np.sum((x - y) ** 2))
    b = np.sqrt(np.sum((y - z) ** 2))
    c = np.sqrt(np.sum((z - x) ** 2))
    sp = (a + b + c) / 2.0

    return np.sqrt(sp * (sp - a) * (sp - b) * (sp - c))


@jit()
def _get_plane_coeff(x, y, z):
    """private method: compute plane coefficients in 3D given three points"""
    a = ((y[1] - x[1]) * (z[2] - x[2]) -
         (z[1] - x[1]) * (y[2] - x[2]))

    b = ((y[2] - x[2]) * (z[0] - x[0]) -
         (z[2] - x[2]) * (y[0] - x[0]))

    c = ((y[0] - x[0]) * (z[1] - x[1]) -
         (z[0] - x[0]) * (y[1] - x[1]))

    d = -(a * x[0] + b * x[1] + c * x[2])
    return a, b, c, d


@jit()
def _get_single_distance(x, y, z, com):
    """private method: returns the distance between a point p and a plane passing through 3 point (x, y, z)"""
    a, b, c, d = _get_plane_coeff(x, y, z)
    distance = a * com[0] + b * com[1] + c * com[2] + d
    distance /= np.sqrt(a ** 2 + b ** 2 + c ** 2)
    return distance


@jit()
def get_surface_volume(verts, faces, com):
    """
    compute the surface and volume of a convex (or start convex object)
    a ration indicating how many points are not fullfilling the star convexity requirement is return,
        if ratio is too big volume approximation should be discarded.
    input: verts (N, 3) mesh nodes coordinates
           faces (M, (3)) list of faces id for each triangle
           com (3) center of mass or any point for which the mesh is approx start convex
    returns: surface, volume (approx), ratio
    """
    surface, volume, com_inside, com_outside = 0, 0, 0, 0
    # for every face
    for face in faces:
        i, j, k = face
        x, y, z = verts[i], verts[j], verts[k]

        # get triangle area and add to total
        area = _get_triangle_area(x, y, z)
        surface += area

        # get piramid volume and add to total volume 
        distance = _get_single_distance(x, y, z, com)
        volume += (distance * area) / 3.0

        # if distance is negative means that com is not in sight of (x, y, z)
        if distance > 0:
            com_inside += 1
        else:
            com_outside += 1

    return surface, volume, com_outside / (com_inside + 1e-16)


def get_exact_volume(mask, voxel_res=(1, 1, 1)):
    """exact volume from mask"""
    return mask.sum() * np.prod(voxel_res)


def standard_score_normalization(coord, com=None):
    """compute the standar score normalization on the coordinates (N, 3)"""
    if com is None:
        com = get_com(coord)
    return (coord - com) / np.std(coord, 0)


@jit()
def _get_edges(faces):
    """private method: compute graph edges from faces (M, (3))"""
    edges = []
    for face in faces:
        for i in range(3):
            edges.append([face[-i], face[-i + 1]])
    return np.array(edges)


def get_adjacency_meshes(faces, verts=None):
    """return adjacency matrix (correct only if input is faces), if verts are supplied the Adjacency is weighted"""
    edges = _get_edges(faces)
    if verts is None:
        weights = np.ones_like(edges[:, 0])
    else:
        weights = np.sqrt(np.sum((verts[edges[:, 0]] - verts[edges[:, 1]]) ** 2, axis=-1))

    # this line works only for meshes!!! (for generic graph to adjacency this would not work)
    A = csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(edges.max() + 1, edges.max() + 1))
    return A


def node_sampling(mode, sample_size, total_nodes):
    if mode == 'grid':
        sampled_nodes = np.linspace(0, total_nodes, sample_size).astype(np.int)
    elif mode == 'random':
        sampled_nodes = np.random.randint(0, total_nodes, size=sample_size).astype(np.int)
    elif mode == 'all':
        sampled_nodes = np.arange(0, total_nodes).astype(np.int)
    else:
        raise NotImplementedError
    return sampled_nodes


def get_sampled_shortest_path(A=None, faces=None, verts=None, num_sampled_points=300, mode='grid'):
    """
    compute the shortest path between random two random points on the surface.
    if mode is grid the points are sampled regularly, else random the points are sampled randomly
    """
    if A is None:
        A = get_adjacency_meshes(faces, verts=verts)

    num_nodes = A.shape[0] - 1
    sampled_nodes = node_sampling(mode, sample_size=num_sampled_points, total_nodes=num_nodes)

    distance_matrix = shortest_path(A, indices=sampled_nodes)
    distance_matrix = distance_matrix[:, sampled_nodes]
    return distance_matrix.ravel()


def get_surface_curvature(verts, faces, normals, radius=1, num_sampled_points=300, mode='grid'):
    """
    compute the shortest path between random two random points on the surface.
    if mode is grid the points are sampled regularly, else random the points are sampled randomly
    """
    num_nodes = verts.shape[0] - 1
    sampled_nodes = node_sampling(mode, sample_size=num_sampled_points, total_nodes=num_nodes)
    x = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    return curvature.discrete_mean_curvature_measure(x, verts[sampled_nodes], radius)


def get_distance_from_com(verts, com, num_sampled_points=None, mode='all'):
    """compute the distances from all (or some) points to the center of mass"""
    num_nodes = verts.shape[0] - 1
    sampled_nodes = node_sampling(mode, sample_size=num_sampled_points, total_nodes=num_nodes)

    distances = np.sqrt(np.sum((verts - com[None, :]) ** 2, axis=-1))

    return distances


def get_histograms(feature, bins):
    """compute the histogram for an harbitrary array (N)"""
    feature[np.isinf(feature)] = 0

    hist, _bins = np.histogram(feature, bins=bins, density=True)
    bins = [_bins.min(), _bins.max(), _bins.shape[0]]
    return hist, bins


class SphHarmFitter:
    """Experimental!!!"""

    def __init__(self, l_max):
        assert l_max >= 0

        l = list(np.arange(0, l_max).astype(np.int))
        m = self._get_m(l)

        self.l = l
        self.m = m
        self.coef = {}

    def __repr__(self):
        """ print components layout"""
        _repr = 'spherical harmonics:\n'
        for l, m in zip(self.l, self.m):
            _repr += f'l: {l}, m: {m}\n'
        return _repr

    @staticmethod
    def _get_m(l):
        m = []
        for _l in l:
            _m_l = []
            for _m in range(-_l, _l + 1):
                _m_l.append(_m)
            m.append(_m_l)
        return m

    def fit(self, data):
        """fit the coefficients from the data (N, 2) [(0, 2*pi), (0, pi)]"""
        coef = {}
        for l, _m in zip(self.l, self.m):
            for m in _m:
                coef[(m, l)] = np.mean(sph_harm(m, l, data[:, 0], data[:, 1]))
        self.coef = coef

    def predict(self, data):
        """evaluate data points with the coefficients"""
        _res = np.zeros(data.shape[0]).astype(np.complex)
        for (m, l), coef in self.coef.items():
            _res += coef * sph_harm(m, l, data[:, 0], data[:, 1])
        return _res

    @staticmethod
    def cartesian2spherical(data_coo):
        r = np.sqrt(np.sum(data_coo ** 2, -1))
        phi = np.arctan2(data_coo[:, 1], data_coo[:, 0])
        phi = np.where(phi > 0, phi, phi + 2 * np.pi)

        theta = np.arccos(data_coo[:, 2] / r)
        return np.stack([r, phi, theta]).T

    @staticmethod
    def spherical2cartesian(data_sph):
        x = data_sph[:, 0] * np.sin(data_sph[:, 1]) * np.cos(data_sph[:, 2])
        y = data_sph[:, 0] * np.sin(data_sph[:, 1]) * np.sin(data_sph[:, 2])
        z = data_sph[:, 0] * np.cos(data_sph[:, 1])
        return np.stack([x, y, z]).T

    @staticmethod
    def normalize(data_coo):
        data_coo_norm = data_coo / np.sqrt(np.sum(data_coo ** 2, -1))[:, None]
        return data_coo_norm

    def compute_power_spectrum(self, verts):
        norm_verts = self.normalize(verts - get_com(verts))
        sph_verts = self.cartesian2spherical(norm_verts)
        self.fit(sph_verts[:, 1:])
        all_coef, all_labels = [], []
        for key, i in self.coef.items():
            if key[0] >= 0:
                all_coef.append(np.absolute(i))
                all_labels.append(key)

        return np.array(all_coef), all_labels


def features_extractor_meshes(mask, verts, faces, normals,
                              param=None):
    # basic features
    if param is None:
        param = {'step_size': 2, 'bins': 100,
                 'com_mode': 'all', 'com_num_sampled_points': 300,
                 'spath_mode': 'grid', 'spath_num_sampled_points': 300,
                 'curv_mode': 'grid', 'curv_num_sampled_points': 300,
                 'radius': 0.5, 'sph_l_max': 5}
    com = get_com(verts)
    mu = get_mu(verts)

    # PCA and inertia
    axis, inv_axis = get_pca_components(verts)
    inertia_tensor, inertia_eigen = get_inertia(mask, mu)

    # transforms

    # surface, volume, convexity
    surface, volume, com_convexity = get_surface_volume(verts, faces, com)

    # distance based features
    com_hist, com_bins = get_histograms(get_distance_from_com(verts,
                                                              com,
                                                              mode=param['com_mode'],
                                                              num_sampled_points=param['com_num_sampled_points']),
                                        param['bins'])
    spath_hist, spath_bins = get_histograms(get_sampled_shortest_path(faces=faces,
                                                                      verts=verts,
                                                                      mode=param['spath_mode'],
                                                                      num_sampled_points=param[
                                                                          'spath_num_sampled_points']),
                                            param['bins'])

    curv_hist, curv_bins = get_histograms(get_surface_curvature(verts,
                                                                faces,
                                                                normals,
                                                                radius=param['radius'],
                                                                mode=param['curv_mode'],
                                                                num_sampled_points=param['curv_num_sampled_points']),
                                          param['bins'])

    sph_spectrum, sph_labels = SphHarmFitter(param['sph_l_max']).compute_power_spectrum(verts)

    features = {'com': com,
                'mu': mu,
                'verts': verts,
                'faces': faces,
                'pca_axis': axis,
                'inv_pca_axis': inv_axis,
                'inertia_tensor': inertia_tensor,
                'inertia_eigen': inertia_eigen,
                'surface': surface,
                'volume': volume,
                'com_convexity': com_convexity,
                'com_hist': com_hist,
                'com_bins': com_bins,
                'spath_hist': spath_hist,
                'spath_bins': spath_bins,
                'curv_hist': curv_hist,
                'curv_bins': curv_bins,
                'sph_spectrum': sph_spectrum,
                'sph_labels': sph_labels}

    return features


def features_extractor(mask, voxel_res=(1, 1, 1), param=None):
    """
    """
    if param is None:
        param = {'step_size': 2,
                 'bins': 100,
                 'com_mode': 'all',
                 'com_num_sampled_points': 300,
                 'spath_mode': 'grid',
                 'spath_num_sampled_points': 300,
                 'curv_mode': 'grid',
                 'curv_num_sampled_points': 300,
                 'radius': 0.5,
                 'sph_l_max': 5}
    verts, faces, normals, values = measure.marching_cubes(mask, step_size=param['step_size'], spacing=voxel_res)
    return features_extractor_meshes(mask, verts, faces, normals, param)
