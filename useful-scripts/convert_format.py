import argparse
import glob
import os

from plantsegtools.utils.edit import crop_image
from plantsegtools.utils.io import smart_load, create_h5, create_tiff


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help='path to segmentation file')
    parser.add_argument("--new-base", type=str, help='optional custom saving directory')
    parser.add_argument("--to-tiff", action='store_true',
                        help='This flag indicates that the segmentation file is tiff.')
    parser.add_argument("--to-h5", action='store_true', help='This flag indicates that the segmentation file is h5.')
    parser.add_argument("--h5-dataset", default='segmentation', help='h5 internal dataset name')
    parser.add_argument("--crop", default=[0, 0, 0, -1, -1, -1], nargs='+', type=int,
                        help='crop the dataset, takes as input a bounding box. eg --crop 10, 0, 0 15, -1, -1.')
    return parser.parse_args()


def format_out_name(path, base=None, ext='.h5'):
    full_base, _ = os.path.splitext(path)
    old_base, name = os.path.split(full_base)
    base = old_base if base is None else base
    out_path = os.path.join(base, f'{name}{ext}')
    return out_path


if __name__ == '__main__':
    args = parse()
    path = args.path
    base = args.new_base
    to_tiff = args.to_tiff
    to_h5 = args.to_h5
    h5_dataset = args.h5_dataset
    crop = args.crop

    paths = glob.glob(path)
    for _path in paths:
        stack, voxel_size = smart_load(_path, key=h5_dataset)
        stack = crop_image(stack, start=crop[:3], end=crop[3:])
        if to_h5:
            out_path = format_out_name(path, base, ext='.h5')
            create_h5(out_path, stack, h5_dataset, voxel_size)

        elif to_tiff:
            out_path = format_out_name(path, base, ext='.tiff')
            create_tiff(out_path, stack, voxel_size)

        else:
            print('please specify in which direction do you want to convert one by using the flag to_tiff or to_h5')
            raise NotImplementedError
