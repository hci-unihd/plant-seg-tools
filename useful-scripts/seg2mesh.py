import argparse
from plantsegtools.meshes.meshes import seg2mesh, seg2mesh_ray
from plantsegtools.meshes.vtkutils import CreateMeshVTK, create_ply
from plantsegtools.utils import TIFF_FORMATS, H5_FORMATS, get_largest_object
import time
import os
import numpy as np
import glob
from datetime import datetime


def parse():
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument("--path", type=str, required=True, help='path to a segmentation file or'
                                                                ' to a directory for batch processing'
                                                                ' of multiple stacks')

    # Optional - path setup
    parser.add_argument("--new-base", type=str, help='optional custom saving directory. '
                                                     'If not given the ply will be saved in the same dir as the source')
    parser.add_argument("--h5-dataset", help='h5 internal dataset name. Default: segmentation',
                        default='segmentation')
    parser.add_argument("--labels", help='List of labels to process. By default the script will process all labels',
                        default=None, nargs='+', type=int)

    # Optional - pipeline parameters
    parser.add_argument('--step-size', help='Step size for the marching cube algorithm, '
                                            'larger steps yield a coarser but faster result.'
                                            ' Default 2.', type=int, default=2)
    parser.add_argument("--crop", default=[0, 0, 0, -1, -1, -1], nargs='+', type=int,
                        help='Crop the dataset, takes as input a bounding box. eg --crop 10, 0, 0 15, -1, -1.')
    parser.add_argument("--voxel-size", default=None, nargs='+', type=float,
                        help='Voxel size of the segmentation stack.'
                             ' By default voxel size is read from the source file,'
                             ' if this is not possible voxel-size is set to [1, 1, 1].')
    parser.add_argument('--min-size', help='Minimum cell size. Default 50.', type=int, default=50)
    parser.add_argument('--max-size', help='Maximum cell size. Default inf.', type=int, default=-1)
    parser.add_argument('--relabel', help='If this argument is passed the pipeline will relabel the segmentation.'
                                          ' This will ensure the contiguity of each segment but will change the '
                                          'labels.', action='store_true')
    parser.add_argument('--check-cc', help='If this argument is passed the pipeline will check if each label is'
                                           ' has a single connected component (cc).'
                                           ' If multiple cc are present only the largest will be processed.',
                        action='store_true', default=False)

    # Optional - mesh processing parameters
    parser.add_argument('--reduction', type=float,
                        help='If reduction > 0 a decimation filter is applied.' ' MaxValue: 1.0 (100%reduction).',
                        default=-.0)
    parser.add_argument('--smoothing', help='To apply a Laplacian smoothing filter.', action='store_true')

    # Optional - multiprocessing
    parser.add_argument('--use-ray', help='If use ray flag is used the multiprocessing flag is managed by ray',
                        action='store_true')
    parser.add_argument('--multiprocessing', help='Define the number of cores to use for parallel processing.'
                                                  ' Default value (-1) will try to parallelize over'
                                                  ' all available processors.', default=-1, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    print(f"[{datetime.now().strftime('%d-%m-%y %H:%M:%S')}] start pipeline setup, parameters: {vars(args)}")

    # Setup path
    if os.path.isfile(args.path):
        all_files = [args.path]
    elif os.path.isdir(args.path):
        all_files = glob.glob(os.path.join(args.path, f'*{TIFF_FORMATS}'))
        all_files += glob.glob(os.path.join(args.path, f'*{H5_FORMATS}'))
    else:
        raise NotImplementedError

    # Optional - path setups
    list_labels = None if args.labels is None else args.labels
    default_prefix = 'meshes'

    # Optional - pipeline parameters
    min_size = 0 if args.min_size < 0 else args.min_size
    max_size = np.inf if args.max_size < 0 else args.max_size
    preprocessing = get_largest_object if args.check_cc else None

    # Optional - multiprocessing
    _seg2mesh = seg2mesh_ray if args.use_ray else seg2mesh
    multiprocessing = args.multiprocessing

    # Setup mesh specific utils

    # Seg2mesh mesh backend is completely independent options availables are
    # 'from plantsegtools.meshes.vtkutils import CreateMeshVTK, create_ply'
    # 'from plantsegtools.meshes.trimeshutils import CreateTriMesh, create_ply'
    # * trimesh support is experimental, is easier to use than vtk but results are worst

    mesh_processor = CreateMeshVTK(reduction=args.reduction, smoothing=args.smoothing)
    file_writer = create_ply

    # main loop
    for i, file_path in enumerate(all_files, 1):
        # just use the same name as original file as base
        base_name = os.path.splitext(os.path.split(file_path)[1])[0]
        base_path = os.path.join(os.path.split(file_path)[0],
                                 f'{default_prefix}_{base_name}') if args.new_base is None else args.new_base

        print(f"[{datetime.now().strftime('%d-%m-%y %H:%M:%S')}]"
              f" start processing file: {os.path.split(file_path)[1]} ({i}/{len(all_files)})")
        timer = time.time()
        _seg2mesh(file_path,
                  mesh_processing=mesh_processor,
                  file_writer=create_ply,
                  base_name=base_name,
                  base_path=base_path,
                  n_process=multiprocessing,
                  step_size=args.step_size,
                  h5_key=args.h5_dataset,
                  voxel_size=args.voxel_size,
                  preprocessing=preprocessing,
                  min_size=min_size,
                  max_size=max_size,
                  idx_list=list_labels,
                  relabel_cc=args.relabel,
                  )

        print(f"[{datetime.now().strftime('%d-%m-%y %H:%M:%S')}]"
              f" process complete in {time.time() - timer: .2f}s,"
              f" number of ply generated {len(glob.glob(os.path.join(base_path, '*.ply')))}")
