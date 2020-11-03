import multiprocessing as mp
import numpy as np
import trimesh
from skimage import measure

from plantsegtools.utils.clean_segmentation import relabel_segmentation
from plantsegtools.utils.io import smart_load
import tqdm
from functools import partial


def _filter_2d_masks(mask):
    _z, _x, _y = np.nonzero(mask)
    # returns True only if mask is not flat in any dimension
    return _z.max() > _z.min() and _x.max() > _x.min() and _y.max() > _y.min()


def create_ply(path, mesh):
    mesh_ply = trimesh.exchange.ply.export_ply(mesh)
    with open(path, "wb") as f:
        # write ply to file
        f.write(mesh_ply)


def mask2mesh(mask, voxel_size=(1.0, 1.0, 1.0), level=0, step_size=2, preprocessing=None, postprocessing=None):
    # apply volume preprocessing
    mask = mask if preprocessing is None else preprocessing(mask, voxel_size)

    # create mesh using marching cubes
    vertx, faces, normals, _ = measure.marching_cubes(mask, level=level, spacing=voxel_size, step_size=step_size)
    mesh = trimesh.Trimesh(vertices=vertx, faces=faces, vertex_normals=normals)

    # apply mesh postprocessing
    mesh = mesh if postprocessing is None else postprocessing(mesh)
    return mesh


def masks_generator(segmentation, min_size=50, max_size=np.inf, idx_list=None, relabel_cc=True, remove_labels=(0, )):
    # relabel segmentation to avoid issues with not connected components (only if no idxlist is provided)
    segmentation = relabel_segmentation(segmentation) if relabel_cc and idx_list is None else segmentation

    if idx_list is None:
        filtered_idx = idx_list
    else:
        # Filter small and large segments
        counts = np.bincount(segmentation.ravel())
        objects_list = list(filter(lambda x: min_size < x[1] < max_size and x[0] not in remove_labels, enumerate(counts)))
        filtered_idx = np.asarray(objects_list)[:, 0]

    # yield idx and masks
    for idx in filtered_idx:
        mask = segmentation == idx

        if _filter_2d_masks(mask):
            yield idx, mask


def _seg2mesh(idx, mask,
              base_name='test_stack',
              base_path='./test-ply/',
              voxel_size=(1.0, 1.0, 1.0),
              preprocessing=None,
              postprocessing=None):

    """basic seg2mesh loop"""
    mesh = mask2mesh(mask,
                     voxel_size=voxel_size,
                     level=0,
                     preprocessing=preprocessing,
                     postprocessing=postprocessing)

    create_ply(f"{base_path}/{base_name}_{idx:05d}.ply", mesh)


def seg2mesh(stack_path,
             base_name='test_stack',
             base_path='./test-ply/',
             n_process=2,
             h5_key='label',
             voxel_size=None,
             preprocessing=None,
             postprocessing=None,
             min_size=50,
             max_size=np.inf,
             idx_list=None,
             relabel_cc=False
             ):

    stack, _voxel_size = smart_load(stack_path, h5_key)
    voxel_size = _voxel_size if voxel_size is None else (1.0, 1.0, 1.0)
    partial_seg2mesh = partial(_seg2mesh,
                               base_name=base_name,
                               base_path=base_path,
                               voxel_size=voxel_size,
                               preprocessing=preprocessing,
                               postprocessing=postprocessing)
    _mask_generator = masks_generator(stack,
                                      min_size=min_size,
                                      max_size=max_size,
                                      idx_list=idx_list,
                                      relabel_cc=relabel_cc)
    if n_process > 1:
        list_idx = list(_mask_generator)
        with mp.Pool(processes=n_process) as executor:
            executor.starmap(partial_seg2mesh, list_idx)
    else:
        [partial_seg2mesh(mask, idx) for mask, idx in tqdm.tqdm(_mask_generator)]


if __name__ == '__main__':
    import time
    timer = time.time()
    sample_path = "/home/lcerrone/datasets/small_samples/sample_ovules.h5"
    seg2mesh(sample_path, max_size=10000, n_process=2)
    print("global timer: ", time.time() - timer)
