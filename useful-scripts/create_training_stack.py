import argparse
from plantsegtools.utils import smart_load, create_h5
import glob
import os


def parse():
    parser = argparse.ArgumentParser(description="Script to start the automated training wizard")
    parser.add_argument("--raw-path", type=str, required=True, help='path to h5 or tiff dataset '
                                                                    'containing the raw image')
    parser.add_argument("--label-path", type=str, required=True, help='path to h5 or tiff dataset '
                                                                      'containing the segmentation image')

    parser.add_argument("--raw-dataset", type=str, default='raw', help='dataset containing the raw image, '
                                                                       'only needed if file is non standard h5')
    parser.add_argument("--label-dataset", type=str, default='label', help='dataset containing the label image, '
                                                                           'only needed if file is non standard h5')

    parser.add_argument("--output-dir", type=str, default=None, help='destination directory for the files,'
                                                                     ' if not given same directory is assumed')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()

    base_path, file_name = os.path.split(args.raw_path)
    base_name, ext = os.path.splitext(file_name)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        base_path = args.output_dir

    out_path = os.path.join(base_path, f'{base_name}_training.h5')

    raw_stack, voxel_size = smart_load(args.raw_path, key=args.raw_dataset)
    segmentation_stack, _ = smart_load(args.label_path, key=args.label_dataset)

    if raw_stack.shape != segmentation_stack.shape:
        print(f"Error! raw and label have different shapes ({raw_stack.shape}, {segmentation_stack.shape})")

    if raw_stack.ndim == 2:
        raw_stack = raw_stack[None, ...]
        segmentation_stack = segmentation_stack[None, ...]

    create_h5(out_path, raw_stack, key='raw', voxel_size=voxel_size, mode='w')
    create_h5(out_path, segmentation_stack, key='label', voxel_size=voxel_size, mode='a')
    print(f"All done, training stack is created in {out_path}")