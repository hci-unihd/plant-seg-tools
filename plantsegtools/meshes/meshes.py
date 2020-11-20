import multiprocessing as mp
import os
from functools import partial

import numpy as np
import tqdm
from skimage import measure

try:
    import ray
except ImportWarning:
    pass

from plantsegtools.utils import filter_2d_masks, smart_load, relabel_segmentation


def mask2mesh(mask, mesh_processing=None, voxel_size=(1.0, 1.0, 1.0), level=0, step_size=2, preprocessing=None):
    # apply volume preprocessing
    mask = mask if preprocessing is None else preprocessing(mask)

    # create mesh using marching cubes
    vertx, faces, normals, _ = measure.marching_cubes(mask, level=level, spacing=voxel_size, step_size=step_size)

    # if no mesh processing is defined returns tuple with marching cubes outputs
    mesh = (vertx, faces, normals) if mesh_processing is None else mesh_processing(vertx, faces, normals)
    return mesh


def idx_generator(segmentation, min_size=50, max_size=np.inf, relabel_cc=True, remove_labels=(0,)):
    # relabel segmentation to avoid issues with not connected components (only if no idxlist is provided)
    segmentation = relabel_segmentation(segmentation) if relabel_cc else segmentation

    # Filter small and large segments
    counts = np.bincount(segmentation.ravel())
    objects_list = list(filter(lambda x: min_size < x[1] < max_size and x[0] not in remove_labels, enumerate(counts)))
    filtered_idx = np.asarray(objects_list)[:, 0]

    return filtered_idx, segmentation


def seg2mesh_shared(stack_path,
                    base_path='./test-ply/',
                    h5_key='label',
                    voxel_size=None,
                    min_size=50,
                    max_size=np.inf,
                    idx_list=None,
                    relabel_cc=False,
                    ):

    segmentation, _voxel_size = smart_load(stack_path, h5_key)
    voxel_size = _voxel_size if voxel_size is None else (1.0, 1.0, 1.0)
    os.makedirs(base_path, exist_ok=True)

    if idx_list is None:
        idx_list, segmentation = idx_generator(segmentation,
                                               min_size=min_size,
                                               max_size=max_size,
                                               relabel_cc=relabel_cc)

        return idx_list, segmentation, voxel_size

    else:
        return idx_list, segmentation, voxel_size


def _seg2mesh(idx, segmentation,
              mesh_processing=None,
              file_writer=None,
              base_name='test_stack',
              base_path='./test-ply/',
              voxel_size=(1.0, 1.0, 1.0),
              step_size=2,
              preprocessing=None):

    """basic seg2mesh loop"""
    # create mask
    mask = segmentation == idx

    # if 2D then ignore
    if not filter_2d_masks(mask):
        return None

    mesh = mask2mesh(mask,
                     mesh_processing=mesh_processing,
                     voxel_size=voxel_size,
                     level=0,
                     step_size=step_size,
                     preprocessing=preprocessing)

    file_writer(f"{base_path}/{base_name}_{idx:05d}.ply", mesh)


def seg2mesh(stack_path,
             mesh_processing=None,
             file_writer=None,
             base_name='test_stack',
             base_path='./test-ply/',
             n_process=1,
             step_size=2,
             h5_key='label',
             voxel_size=None,
             preprocessing=None,
             min_size=50,
             max_size=np.inf,
             idx_list=None,
             relabel_cc=False,
             ):
    idx_list, segmentation, voxel_size = seg2mesh_shared(stack_path,
                                                         base_path=base_path,
                                                         h5_key=h5_key,
                                                         voxel_size=voxel_size,
                                                         min_size=min_size,
                                                         max_size=max_size,
                                                         idx_list=idx_list,
                                                         relabel_cc=relabel_cc,
                                                         )

    partial_seg2mesh = partial(_seg2mesh,
                               mesh_processing=mesh_processing,
                               file_writer=file_writer,
                               segmentation=segmentation,
                               base_name=base_name,
                               base_path=base_path,
                               voxel_size=voxel_size,
                               step_size=step_size,
                               preprocessing=preprocessing
                               )

    if n_process > 1:
        with mp.Pool(processes=n_process) as executor:
            executor.map(partial_seg2mesh, idx_list)
    else:
        [partial_seg2mesh(idx) for idx in tqdm.tqdm(idx_list)]


def seg2mesh_ray(stack_path,
                 mesh_processing=None,
                 file_writer=None,
                 base_name='test_stack',
                 base_path='./test-ply/',
                 h5_key='label',
                 step_size=2,
                 voxel_size=None,
                 preprocessing=None,
                 min_size=50,
                 max_size=np.inf,
                 idx_list=None,
                 relabel_cc=False,
                 ):

    idx_list, segmentation, voxel_size = seg2mesh_shared(stack_path,
                                                         base_path=base_path,
                                                         h5_key=h5_key,
                                                         voxel_size=voxel_size,
                                                         min_size=min_size,
                                                         max_size=max_size,
                                                         idx_list=idx_list,
                                                         relabel_cc=relabel_cc,
                                                         )

    partial_seg2mesh = partial(_seg2mesh,
                               mesh_processing=mesh_processing,
                               file_writer=file_writer,
                               base_name=base_name,
                               base_path=base_path,
                               voxel_size=voxel_size,
                               step_size=step_size,
                               preprocessing=preprocessing,
                               )

    ray.init()
    segmentation_id = ray.put(segmentation)

    @ray.remote(num_cpus=1)
    def remote_seg2mesh(idx, _segmentation_id):
        return partial_seg2mesh(idx, _segmentation_id)

    tasks = [remote_seg2mesh.remote(idx, segmentation_id) for idx in idx_list]
    ray.get(tasks)
    ray.shutdown()


if __name__ == '__main__':
    import time
    from plantsegtools.meshes.vtkutils import CreateMeshVTK, create_ply

    sample_path = "/home/lcerrone/datasets/small_samples/sample_ovules.h5"
    mesh_processor = CreateMeshVTK()

    for i in [1, 2]:
        timer = time.time()
        seg2mesh(sample_path, file_writer=create_ply, mesh_processing=mesh_processor, n_process=i)
        print(f"{i} global timer: ", time.time() - timer)

    timer = time.time()
    seg2mesh_ray(sample_path, file_writer=create_ply, mesh_processing=mesh_processor)
    print(f"ray global timer: ", time.time() - timer)
