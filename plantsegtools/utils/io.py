import tifffile
import h5py
import warnings


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


def load_h5(path, key):
    with h5py.File(path, 'r') as f:
        file = f[key][...]
        voxel_size = read_h5_voxel_size(f, key)

    return file, voxel_size


def load_tiff(path):
    file = tifffile.imread(path)
    voxel_size = read_tiff_voxel_size(path)
    return file, voxel_size


def load_lif():
    pass


def create_h5(path, stack, key, voxel_size=(1.0, 1.0, 1.0), mode='a'):
    with h5py.File(path, mode) as f:
        f.create_dataset(key, data=stack, compression='gzip')
        # save voxel_size
        f[key].attrs['element_size_um'] = voxel_size


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

