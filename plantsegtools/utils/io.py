import tifffile
import h5py
import warnings
import os

TIFF_FORMATS = ['.tiff', '.tif']
H5_FORMATS = ['.h5', '.hdf']
LIF_FORMATS = ['.lif']


def read_tiff_voxel_size(file_path):
    """
    Implemented based on information found in https://pypi.org/project/tifffile
    """

    def _xy_voxel_size(tags, key):
        assert key in ['XResolution', 'YResolution']
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return 1.

    with tifffile.TiffFile(file_path) as tiff:
        image_metadata = tiff.imagej_metadata
        if image_metadata is not None:
            z = image_metadata.get('spacing', 1.)
        else:
            # default voxel size
            z = 1.

        tags = tiff.pages[0].tags
        # parse X, Y resolution
        y = _xy_voxel_size(tags, 'YResolution')
        x = _xy_voxel_size(tags, 'XResolution')
        # return voxel size
        return [z, y, x]


def read_h5_voxel_size_file(file_path, h5key):
    with h5py.File(file_path, "r") as f:
        return read_h5_voxel_size(f, h5key)


def read_h5_voxel_size(f, h5key):
    ds = f[h5key]

    # parse voxel_size
    if 'element_size_um' in ds.attrs:
        voxel_size = ds.attrs['element_size_um']
    else:
        warnings.warn('Voxel size not found, returning default [1.0, 1.0. 1.0]', RuntimeWarning)
        voxel_size = [1.0, 1.0, 1.0]

    return voxel_size


def load_h5(path, key, slices=None, safe_mode=False):

    with h5py.File(path, 'r') as f:
        if key is None:
            key = list(f.keys())[0]

        if safe_mode and key not in list(f.keys()):
            return None, (1, 1, 1)

        if slices is None:
            file = f[key][...]
        else:
            file = f[key][slices]

        voxel_size = read_h5_voxel_size(f, key)

    return file, voxel_size


def load_tiff(path):
    file = tifffile.imread(path)
    voxel_size = read_tiff_voxel_size(path)
    return file, voxel_size


def load_lif():
    raise NotImplementedError


def smart_load(path, key=None, default=load_tiff):
    _, ext = os.path.splitext(path)
    if ext in H5_FORMATS:
        return load_h5(path, key)

    elif ext in TIFF_FORMATS:
        return load_tiff(path)

    elif ext in LIF_FORMATS:
        return load_lif(path)

    else:
        print(f"No default found for {ext}, reverting to default loader")
        return default(path)


def create_h5(path, stack, key, voxel_size=(1.0, 1.0, 1.0), mode='a'):
    with h5py.File(path, mode) as f:
        f.create_dataset(key, data=stack, compression='gzip')
        # save voxel_size
        f[key].attrs['element_size_um'] = voxel_size


def del_h5_key(path, key, mode='a'):
    with h5py.File(path, mode) as f:
        if key in f:
            del f[key]
            f.close()


def rename_h5_key(path, old_key, new_key, mode='r+'):
    ''' Rename the 'old_key' dataset to 'new_key' '''
    with h5py.File(path, mode) as f:
        if old_key in f:
            f[new_key] = f[old_key]
            del f[old_key]
            f.close()


def rename_h5_attr(path, key, old_attr, new_attr, mode='r+'):
        ''' Rename the attribute of dataset 'key' from 'old_attr' to 'new_attr'  '''
        with h5py.File(path, mode) as f:
            pass
# http://api.h5py.org/h5a.html#h5py.h5a.rename
# h5py.h5a.rename(myfile.id, b"name", b"newname")

def create_tiff(path, stack, voxel_size):
    # taken from: https://pypi.org/project/tifffile docs
    z, y, x = stack.shape
    stack.shape = 1, z, 1, y, x, 1  # dimensions in TZCYXS order
    spacing, y, x = voxel_size
    resolution = (1. / x, 1. / y)
    # Save output results as tiff
    tifffile.imsave(path,
                    data=stack,
                    dtype=stack.dtype,
                    imagej=True,
                    resolution=resolution,
                    metadata={'axes': 'TZCYXS', 'spacing': spacing, 'unit': 'um'})
