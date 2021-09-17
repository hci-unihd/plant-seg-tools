import argparse
import csv

import numpy as np
from numba import njit, prange
from numba.core import types
from numba.typed import Dict
from scipy.ndimage import zoom
from skimage.segmentation import find_boundaries

from plantsegtools.utils.io import smart_load, create_h5


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stain", type=str, help='path to stain file')
    parser.add_argument("--reporter", type=str, help='path to reporter file')
    parser.add_argument("--seg", type=str, help='path to segmentation file')
    parser.add_argument("--out", type=str, help='path to output file')
    parser.add_argument("--csv", type=str, help='path to csv file')
    parser.add_argument("--flip", default=False, help='if to flip MGX stack', action='store_true')
    return parser.parse_args()


@njit(parallel=True)
def _mapping2image(in_image, out_image, mappings):
    shape = in_image.shape
    for i in prange(0, shape[0]):
        for j in prange(0, shape[1]):
            for k in prange(0, shape[2]):
                out_image[i, j, k] = mappings[in_image[i, j, k]]

    return out_image


def mapping2image(in_image, mappings, data_type='int64'):
    value_type = types.int64 if data_type == 'int64' else types.float64

    numba_mappings = Dict.empty(key_type=types.int64,
                                value_type=value_type)
    numba_mappings.update(mappings)
    numba_mappings[0] = 0

    out_image = np.zeros_like(in_image).astype(data_type)
    out_image = _mapping2image(in_image, out_image, numba_mappings)
    return out_image


def update_labels_mapping(seg, csv_ids, csv_labels):
    seg_u = np.unique(seg)
    labels = np.zeros_like(seg_u)
    cell_feature_mapping = create_features_mapping(seg_u, labels)
    for _ids, _lab in zip(csv_ids, csv_labels):
        cell_feature_mapping[_ids] = _lab
    return cell_feature_mapping


def map_cell_features2segmentation(segmentation, cell_ids, cell_feature):
    cell_feature_mapping = update_labels_mapping(segmentation, cell_ids, cell_feature)
    features_image = mapping2image(segmentation, cell_feature_mapping)
    return features_image


def create_features_mapping(features_ids, features):
    mapping = {}
    for key, value in zip(features_ids, features):
        mapping[key] = value
    return mapping


def import_labels_csv(path, csv_columns=('cell_ids', 'cell_labels')):
    cell_ids, cell_labels = [], []
    with open(path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=csv_columns)
        reader = list(reader)
        for row in reader[1:]:
            cell_ids.append(row[csv_columns[0]])
            cell_labels.append(row[csv_columns[1]])

    return np.array(cell_ids, dtype='int32'), np.array(cell_labels, dtype='int32')


def main():
    args = parse()

    # copy stain
    raw, voxel_size = smart_load(args.stain)
    raw_shape = raw.shape
    create_h5(args.out, raw.astype('float32'), 'nuclei_stain', voxel_size=voxel_size, mode='w')

    if args.reporter is not None:
        stack, _ = smart_load(args.reporter)
        scale = np.array(raw_shape) / np.array(stack.shape)
        if np.prod(scale) != 1.:
            stack = zoom(stack, scale, order=0)
        create_h5(args.out, stack.astype('float32'), 'nuclei_reporter', voxel_size=voxel_size, mode='a')
        stack = np.stack([raw, stack], axis=3)
        create_h5(args.out, stack.astype('float32'), 'raw_merged', voxel_size=voxel_size, mode='a')

    if args.seg is not None:
        stack, _ = smart_load(args.seg)

        if args.flip:
            stack = stack[:, ::-1, :]

        scale = np.array(raw_shape) / np.array(stack.shape)
        if np.prod(scale) != 1.:
            stack = zoom(stack, scale, order=0)

        boundary = find_boundaries(stack)
        stack[boundary] = 0

        mask = stack != 0

        create_h5(args.out, stack.astype('uint32'), 'segmentation', voxel_size=voxel_size, mode='a')
        create_h5(args.out, mask.astype('uint32'), 'segmentation_mask', voxel_size=voxel_size, mode='a')

        if args.csv is not None:
            csv_ids, csv_labels = import_labels_csv(args.csv)
            atlas = map_cell_features2segmentation(stack, csv_ids, csv_labels)
            create_h5(args.out, atlas.astype('uint32'), 'atlas', voxel_size=voxel_size, mode='a')


if __name__ == '__main__':
    main()
