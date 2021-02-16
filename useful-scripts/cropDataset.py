import h5py
import os
import glob
import argparse
from plantsegtools.utils import H5_FORMATS


def parse_crop(crop_str):
    '''Return a tuple with a slice object from a string (ex. "[:, 0:620, 420:1750]") '''
    crop_str = crop_str.replace('[', '').replace(']', '')
    return tuple(
        (slice(*(int(i) if i else None for i in part.strip().split(':'))) if ':' in part else int(part.strip())) for
        part in crop_str.split(','))


def parse():
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument("--path", type=str, required=True, help='path to h5 file or'
                                                                ' to a directory for batch processing'
                                                                ' of multiple files')
    parser.add_argument('--crop', type=str, help='[ZYX] cropping to apply (e.g. "[:, 0:620, 420:1750]")', default='[:,:,:]', required=False)
    return parser.parse_args()


class get3dds():
    ''' Class used by the h5 walker function to store the names of all 3d datasets and of all others'''
    def __init__(self):
        self.sets3d = []
        self.sets_n3d = []

    def __call__(self, name, node):
        if isinstance(node, h5py.Dataset):
            if len(node.shape) > 2:
                self.sets3d.append(name)
            else:
                self.sets_n3d.append(name)
        return None


if __name__ == '__main__':
    '''Crop all >=3D datasets of  an h5 file, leaves other untouched, saves a copy.'''
    args = parse()

    # Setup input path
    if os.path.isfile(args.path):
        all_files = [args.path]
    elif os.path.isdir(args.path):
        all_files = glob.glob(os.path.join(args.path, f'*{H5_FORMATS}'))
    else:
        raise NotImplementedError

    for i, file_path in enumerate(all_files, 1):
        # Prep IO
        path = os.path.dirname(file_path)
        fname = os.path.splitext(os.path.basename(file_path))[0]
        outfname = f"{fname}_crop.h5"
        outpath = os.path.join(path, outfname)

        print(f"Processing {fname}.h5 ({i}/{len(all_files)})")
        with h5py.File(file_path, 'r') as f:
            with h5py.File(outpath, 'w') as fc:
                #  Traverse the file hierarchy and  save names of 3D datasets and the others
                d3ds = get3dds()
                f.visititems(d3ds)
                for ds in d3ds.sets3d:
                    # Process the 3D datasets --> crop
                    if len(f[ds].shape) == 3:
                        crop = parse_crop(args.crop)
                    else:
                        crop = parse_crop(args.crop)
                        crop = (slice(None, None, None), *crop)
                    crop_ds = f[ds][crop]
                    # store the attributes
                    _temp = {}
                    for k, v in f[ds].attrs.items():
                        _temp[k] = v
                    # Write cropped dataset to outfile
                    fc.create_dataset(ds, data=crop_ds, compression='gzip')
                    # reassign the attributes
                    for k, v in _temp.items():
                        fc[ds].attrs[k] = v
                for ds in d3ds.sets_n3d:
                    # For the not 3D datasets, copy them to the outfile
                    # store the attributes
                    _temp = {}
                    for k, v in f[ds].attrs.items():
                        _temp[k] = v
                    # Copy dataset to outfile
                    fc.create_dataset(ds, data=f[ds], compression='gzip')
                    # reassign the attributes
                    for k, v in _temp.items():
                        fc[ds].attrs[k] = v
    i += 1
