import concurrent.futures

import numpy as np
import trimesh
from skimage import measure

from plantsegtools.utils.clean_segmentation import relabel_segmentation
from plantsegtools.utils.io import smart_load


def filter_2d_masks(mask):
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
    mesh = trimesh.Trimesh(vertices=vertx, faces=faces)#, vertex_normals=normals)

    # apply mesh postprocessing
    mesh = mesh if postprocessing is None else postprocessing(mesh)
    return mesh


def masks_generator(segmentation, min_size=50, max_size=np.inf, relabel_cc=True):
    # relabel segmentation to avoid issues with not connected components
    segmentation = relabel_segmentation(segmentation) if relabel_cc else segmentation

    # Filter small and large segments
    all_idx, counts = np.unique(segmentation, return_counts=True)
    filtered_idx = filter(lambda x: min_size < x[1] < max_size, zip(all_idx, counts))

    # yield idx and masks
    for idx, _ in filtered_idx:
        mask = segmentation == idx

        if filter_2d_masks(mask):
            yield idx, mask


def seg2mesh(stack_path,
             save_path='/home/lcerrone/testply/',
             h5_key='label',
             preprocessing=None,
             postprocessing=None,
             min_size=50,
             max_size=np.inf,
             relabel_cc=True
             ):

    def _seg2mesh(args):
        idx, mask = args
        mesh = mask2mesh(mask,
                         voxel_size=voxel_size,
                         level=0,
                         preprocessing=preprocessing,
                         postprocessing=postprocessing)

        create_ply(f"{save_path}/test{idx}.ply", mesh)

    stack, voxel_size = smart_load(stack_path, h5_key)
    [_seg2mesh(i) for i in masks_generator(stack, min_size=min_size, max_size=max_size, relabel_cc=relabel_cc)]


def _seg2mesh_mp(args):
    idx, mask = args
    save_path = '/home/lcerrone/testply/'
    mesh = mask2mesh(mask,
                     voxel_size=(1.0, 1.0, 1.0),
                     level=0,
                     preprocessing=None,
                     postprocessing=None)
    create_ply(f"{save_path}/test{idx}.ply", mesh)


def seg2mesh_mp(stack_path,
                save_path='/home/lcerrone/testply/',
                h5_key='label',
                preprocessing=None,
                postprocessing=None,
                min_size=50,
                max_size=np.inf,
                relabel_cc=True
                ):

    stack, voxel_size = smart_load(stack_path, h5_key)
    list_idx = list(masks_generator(stack, min_size=min_size, max_size=max_size, relabel_cc=relabel_cc))
    with concurrent.futures.ProcessPoolExecutor(2) as executor:
        executor.map(_seg2mesh_mp, list_idx)


if __name__ == '__main__':
    import time
    timer = time.time()
    sample_path = "/home/lcerrone/datasets/small_samples/sample_ovules.h5"
    #seg2mesh(sample_path, max_size=100000)
    seg2mesh_mp(sample_path, max_size=100000)

    print(time.time() - timer)
