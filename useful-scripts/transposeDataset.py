import h5py
import os
import glob
import argparse
import numpy as np
# from plantsegtools.utils.io import rename_h5_key
from plantsegtools.utils import  H5_FORMATS

def parse():
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument("--path", type=str, required=True, help='path to h5 file or'
                                                                ' to a directory for batch processing'
                                                                ' of multiple files')
    parser.add_argument("--new_order", default=None, nargs='+', type=int,
                        help='New axis order 0 1 2 for ZYX -->  0 2 1 for  ZXY ')
    return parser.parse_args()


if __name__ == '__main__':
    '''Transpose  axes of all datasets of  an h5 file'''
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
        with h5py.File(file_path, 'r+') as data:
            for key in data.keys():
                if len(data[key].shape) == 3:
                    print(f"{key}: {data[key].shape}", end="", flush=True)
                    # store the attributes
                    _temp ={}
                    for k,v in data[key].attrs.items():
                        _temp[k] = v
                    # Transpose the data
                    t_data = np.transpose(data[key], (args.new_order[0], args.new_order[1], args.new_order[2]))
                    # erase the dataset
                    del data[key]
                    # create the transposed one
                    data.create_dataset(key,  data=t_data, compression='gzip')
                    print(f" --> {data[key].shape}")
                    # reassign the attributes
                    for k,v in _temp.items():
                        data[key].attrs[k] = v
    i += 1
