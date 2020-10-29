import numpy as np
from scipy.ndimage import zoom
from skimage.filters import median
from skimage.morphology import ball, disk

try:
    from vigra.filters import gaussianSmoothing
except:
    from skimage.filters import gaussian as gaussianSmoothing


def crop_image(image, start=(0, 0, 0), end=(-1, -1, -1)):
    assert image.ndim == 2 or image.ndim == 3
    assert len(start) == len(end) == image.ndim
    _slice = tuple([slice(s, e) for s, e in zip(start, end)])
    return image[_slice]


def safe_cast(image, out_type):
    possible_float = ['float', 'float32', 'float64']
    possible_int = ['int', 'int8', 'int16', 'int32', 'int64',
                    'uint8', 'uint16', 'uint32', 'uint64']
    in_type = image.dtype
    assert out_type in possible_float + possible_int
    assert in_type in possible_float + possible_int

    if out_type in possible_float:
        if in_type in possible_float:
            return image.astype(out_type)
        elif in_type in possible_int:
            n_image = _0to1(image)
            return n_image.astype(out_type)
    else:
        # This covers all case where the out is in possible int
        n_image = _0to1(image)
        n_image *= np.iinfo(out_type).max
        return n_image.astype(out_type)


def _0to1(image, eps=1e-16):
    return (image - image.min()) / (image.max() - image.min() + eps)


def _z_score(image, std=None, eps=1e-16):
    std = np.std(image) if std is None else std
    mean = np.mean(image)
    return (image - mean) / (std + eps)


def normalize_image(image, mode='0to1'):
    assert mode in ['0to1', '-1to1', 'z-score', 'zero-mean']
    _image = image.astype(np.flat32)
    if mode == '0to1':
        return _0to1(_image)
    elif mode == '-1to1':
        return 2 * _0to1(_image) - 1
    elif mode == 'z-score':
        return _z_score(_image)
    elif mode == 'zero-mean':
        return _z_score(_image, std=1, eps=0)
    else:
        raise NotImplementedError


def change_resolution(image, new_resolution, old_resolution, order=2):
    assert len(new_resolution) == len(old_resolution)
    factor = [n/o for n, o in zip(new_resolution, old_resolution)]

    if np.array_equal(factor, len(factor) * [1]):
        return image
    else:
        return zoom(image, zoom=factor, order=order)


def change_segmentation_resolution(image, new_resolution, old_resolution):
    return change_resolution(image, new_resolution, old_resolution, order=0)


def median_filter(image, radius):
    if image.shape[0] == 1:
        shape = image.shape
        median_image = median(image[0], disk(radius))
        return median_image.reshape(shape)
    else:
        return median(image, ball(radius))


def gaussian_filter(image, sigma):
    max_sigma = (np.array(image.shape) - 1) / 3
    sigma = np.minimum(max_sigma, np.ones(max_sigma.ndim) * sigma)
    return gaussianSmoothing(image, sigma)
