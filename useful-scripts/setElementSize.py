import h5py
import os
import glob
import argparse
# from plantsegtools.utils.io import rename_h5_key
from plantsegtools.utils import  H5_FORMATS

def parse():
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument("--path", type=str, required=True, help='path to h5 file or'
                                                                ' to a directory for batch processing'
                                                                ' of multiple files')
    parser.add_argument("--voxel-size", default=None, nargs='+', type=float,
                        help='Voxel size [Z, Y, X] in um')
    return parser.parse_args()


if __name__ == '__main__':
    '''Set the 'element_size_um'  attribute in all datasets of  an h5 file'''
    args = parse()

    # Setup input path
    if os.path.isfile(args.path):
        all_files = [args.path]
    elif os.path.isdir(args.path):
        all_files = glob.glob(os.path.join(args.path, f'*{H5_FORMATS}'))
    else:
        raise NotImplementedError

    for i, file_path in enumerate(all_files, 1):
        print(f"Processing {os.path.split(file_path)[1]} ({i}/{len(all_files)})")
        data = h5py.File(file_path, 'a')
        for key in data.keys():
            data[key].attrs['element_size_um'] = (args.voxel_size[0], args.voxel_size[1], args.voxel_size[2])
        i += 1
