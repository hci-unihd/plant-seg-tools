import argparse
from plantsegtools.meshes.meshes import seg2mesh, seg2mesh_ray
from plantsegtools.meshes.vtkutils import CreateMeshVTK, create_ply
import time
import os
import numpy as np


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help='path to segmentation file')
    parser.add_argument("--new-base", type=str, help='optional custom saving directory')
    parser.add_argument("--h5-dataset", default='segmentation', help='h5 internal dataset name')
    parser.add_argument('--step-size', help='Step size for the marching cube algorithm, '
                                            'larger steps yield a coarser but faster result. Default 2 (voxel).',
                        default=2, required=False)

    parser.add_argument('--use-ray', help='If use ray. --multiprocessing flag is ignored and'
                                          ' resources are managed by ray',
                        action='store_true')
    # Multiprocessing
    parser.add_argument('--multiprocessing', help='Define the number of cores to use for parallel processing.',
                        required=False, default=-1, type=int)
    parser.add_argument("--crop", default=[0, 0, 0, -1, -1, -1], nargs='+', type=int,
                        help='crop the dataset, takes as input a bounding box. eg --crop 10, 0, 0 15, -1, -1.')

    # Mesh processing
    parser.add_argument('--reduction', type=float,
                        help='If reduction > 0 a decimation filter is applied.' ' MaxValue: 1.0 (100%reduction).',
                        default=-.0, required=False)
    parser.add_argument('--smoothing', help='To apply a Laplacian smoothing filter.', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    path = args.path
    new_base = args.new_base
    dataset = args.h5_dataset
    step_size = args.step_size
    use_ray = args.use_ray
    multiprocessing = args.multiprocessing
    crop = args.crop

    # Seg2mesh mesh backend is completely independent options availables are
    # 'from plantsegtools.meshes.vtkutils import CreateMeshVTK, create_ply'
    # 'from plantsegtools.meshes.trimeshutils import CreateTriMesh, create_ply'
    # * trimesh support is experimental, is easier to use than vtk but results are worst

    mesh_processor = CreateMeshVTK(reduction=args.reduction, smoothing=args.smoothing)
    file_writer = create_ply

    # just use the same name as original file
    base_name = os.path.splitext(os.path.split(path)[-1])[0]

    timer = time.time()
    _seg2mesh = seg2mesh_ray if use_ray else seg2mesh
    _seg2mesh(path,
              mesh_processing=mesh_processor,
              file_writer=create_ply,
              base_name=base_name,
              base_path=new_base,
              n_process=multiprocessing,
              step_size=step_size,
              h5_key=dataset,
              voxel_size=None,
              preprocessing=None,
              min_size=50,
              max_size=np.inf,
              idx_list=None,
              relabel_cc=False,
              )

    print(f"global timer: ", time.time() - timer)
