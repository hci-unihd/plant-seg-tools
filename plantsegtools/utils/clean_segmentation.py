import numpy as np
from scipy import ndimage
from skimage import measure
import numba


def relabel_segmentation(seg):
    """
    Relabel contiguously a segmentation image, non-touching instances with same id will be relabeled differently.
    To be noted that measure.label is different from ndimage.label
    Example:
        x = np.zeros((5, 5))
        x[:2], x[2], x[3:]= 1, 2, 1
        print(x)
        print(ndimage.label(x)[0])
        print(measure.label(x))

        x = np.zeros((5, 5))
        x[:2], x[2], x[3:]= 1, 0, 1
        print(x)
        print(ndimage.label(x)[0])
        print(measure.label(x))
    """
    return measure.label(seg)


@numba.njit()
def _replace(arr, needle, replace=0):
    """
    arr must a 1d array and needles must be a set.

    Code sourced from:
    https://stackoverflow.com/questions/43942943/set-specific-values-to-zero-in-numpy-array
    """
    needles = set(needle)
    for idx in range(arr.size):
        if arr[idx] in needles:
            arr[idx] = replace
    return arr


def remove_listed_objects(segmentation, objects_list, not_listed_mode=False, background_label=0, inplace=False,
                          unique=None):
    """
    Remove listed indices from a segmentation image.
    segmentation: integer array
    object list: list of indices to be removed from the segmentation image
    not_listed_mode: if True only listed object are going to be preserved
    """
    _seg = segmentation if inplace else np.copy(segmentation)

    if not_listed_mode:
        labels = np.unique(_seg) if unique is None else unique
        objects_list = list(filter(lambda _obj: _obj not in objects_list,
                                   labels))

    objects_list = np.asarray(objects_list)
    _seg = _replace(_seg.ravel(), objects_list, background_label).reshape(_seg.shape)
    return _seg


def segment_size_filter(segmentation, min_size=0,
                        max_size=np.inf,
                        voxel_size=(1.0, 1.0, 1.0),
                        background_label=0,
                        inplace=False):
    """
    filter segments smaller of min_size or larger than max_size.
    If voxel_size is defined then min_size and max_size will have the same units.
    """
    _seg = segmentation if inplace else np.copy(segmentation)

    min_size *= np.prod(voxel_size)
    max_size *= np.prod(voxel_size)

    counts = np.bincount(_seg.ravel())
    objects_list = list(filter(lambda x: x[1] != 0 and x[1] < min_size or x[1] > max_size, enumerate(counts)))
    objects_list = np.asarray(objects_list)[:, 0]

    _seg = _replace(_seg.ravel(), objects_list, background_label).reshape(_seg.shape)
    return _seg


def get_largest_object(mask):
    """Returns largest connected components"""
    # ~2x faster than clean_object(obj)
    # relabel connected components
    labels, numb_components = ndimage.label(mask)
    assert numb_components > 0  # assume at least 1 CC
    if numb_components == 1:
        return mask
    else:
        return labels == np.argmax(np.bincount(labels.flat)[1:]) + 1


def _filter_2d_masks(mask):
    ### Legacy version
    _z, _x, _y = np.nonzero(mask)
    # returns True only if mask is not flat in any dimension
    return abs(_z.max() - _z.min()) > 1 and abs(_x.max() - _x.min()) > 1 and abs(_y.max() - _y.min()) > 1


def filter_2d_masks(mask):
    _z, _x, _y = np.nonzero(mask)
    # Check if any dimension is flat
    if (abs(_z.max() - _z.min()) <= 1 or
            abs(_x.max() - _x.min()) <= 1 or
            abs(_y.max() - _y.min()) <= 1):
        return False

    # check if the segment has any thickness
    bbox = mask[_z.min():_z.max(), _x.min():_x.max(), _y.min():_y.max()]
    bin_sum = np.sum(ndimage.binary_erosion(bbox, structure=np.ones((2, 2, 2)).astype(np.int)))

    return True if bin_sum > 0 else False

