import argparse
import glob
import re
import h5py
import os
from plantsegtools.utils.edit import parse_crop


def main():
    parser = argparse.ArgumentParser(description='Copy a dataset from one H5 to another. Files with identical time point in names will be matched.')
    parser.add_argument('--source_dir', type=str, help='Path to the source files', required=True)
    parser.add_argument('--dest_dir', type=str, help='Path to the destination files', required=True)
    parser.add_argument('--crop', type=str, help='[ZYX] cropping to apply (e.g. "[:, 0:620, 420:1750]")', default='[:,:,:]', required=False)
    parser.add_argument('--source_dataset', type=str, help='Name of the dataset to import', default='Data', required=False)
    parser.add_argument('--dest_dataset', type=str, help='Name of the dataset at destination', default='Data', required=False)

    args = parser.parse_args()
    for source_file in glob.glob(os.path.join(args.source_dir, '*.h5')):
        filename_source = os.path.split(source_file)[1]
        tp_source = re.search('[Tt](\d{1,})', filename_source).group(1)

        for dest_file in glob.glob(os.path.join(args.dest_dir, '*.h5')):
            filename_dest = os.path.split(dest_file)[1]
            tp_dest = re.search('[tT](\d{1,})', filename_dest).group(1)
            if tp_source == tp_dest:
                print(f"{filename_source}/['{args.source_dataset}'] ---> {filename_dest}/['{args.dest_dataset}']")
                crop = parse_crop(args.crop)
                with h5py.File(source_file, 'r+') as source:
                    d_source = source[args.source_dataset][crop]
                with h5py.File(dest_file, 'a') as dest:
                    if args.dest_dataset in dest:
                        dest[args.dest_dataset][:]= d_source
                    else:
                        dest.create_dataset('/' + args.dest_dataset, data=d_source, compression='gzip')
                dest.close()
                source.close()


if __name__ == '__main__':
    print("Starting copyDataset")
    main()
    print("Done!")
