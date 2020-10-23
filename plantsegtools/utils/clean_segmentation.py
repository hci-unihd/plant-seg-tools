import numpy as np
from scipy import ndimage
from skimage import measure


def relabel_segmentation(seg):
    return measure.label(seg)


def remove_listed_object(seg, object_list, not_listed_mode=False, background_label=0):

    if not_listed_mode:
        labels = np.unique(seg)
        object_list = list(filter(lambda _obj: _obj not in object_list, labels))

    for obj in object_list:
        seg[seg == obj] = background_label

    return seg


def remove_small_object(seg, min_size=0, units='voxel', background_label=0):
    """
    do not use, use skimage instead
    """
    if units != 'voxel':
        raise NotImplementedError

    objects, counts = np.unique(seg, return_counts=True)
    objects = objects[list(map(lambda _count: _count > min_size, counts))]

    for obj in objects:
        seg[seg == obj] = background_label

    return seg


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
