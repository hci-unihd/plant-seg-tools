import argparse
import time
import warnings

import numpy as np
from scipy.ndimage import zoom

from plantsegtools.evaluation.rand import adapted_rand
from plantsegtools.evaluation.voi import voi
from plantsegtools.utils.io import smart_load

# Add new metrics if needed
metrics = {"voi": (voi, 'Split and Merge Error'),
           "adapted_rand": (adapted_rand, 'AdaptedRand Error')}


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--path-seg', type=str, required=False, help='path to segmentation file')
    parser.add_argument('-ds', '--dataset-seg', type=str, default='segmentation',
                        required=False, help='if h5 contains dataset name to segmentation')
    parser.add_argument('-g', '--path-gt', type=str, required=False, help='path to ground-truth file')
    parser.add_argument('-dg', '--dataset-gt', type=str, default='label',
                        required=False, help='if h5 contains dataset name to segmentation')
    return parser.parse_args()


def run_evaluation(gt_array, seg_array, remove_background=True):
    timer = - time.time()
    # Check for problems in data types
    # double check for type and sign to allow a bit of slack in using _
    # int for segmentation and not only uint)
    if not np.issubdtype(seg_array.dtype, np.integer):
        return None

    if not np.issubdtype(gt_array.dtype, np.integer):
        warnings.warn("Ground truth is not an integer array")
        return None

    if np.any(seg_array < 0):
        warnings.warn("Found negative indices, segmentation must be positive")
        return None

    if np.any(gt_array < 0):
        warnings.warn("Found negative indices, ground truth must be positive")
        return None

    # Cast into uint3232
    seg_array = seg_array.astype(np.uint32)
    gt_array = gt_array.astype(np.uint32)

    # Resize segmentation to gt size for apple to apple comparison in the scores
    if seg_array.shape != gt_array.shape:
        print("- Segmentation shape:", seg_array.shape,
              "Ground truth shape: ", gt_array.shape)

        print("- Shape mismatch, trying to fixing it")
        factor = tuple([g_shape / seg_shape for g_shape, seg_shape in zip(gt_array.shape, seg_array.shape)])
        seg_array = zoom(seg_array, factor, order=0).astype(np.uint32)

    if remove_background:
        print("- Removing background")
        mask = gt_array != 0
        gt_array = gt_array[mask].ravel()
        seg_array = seg_array[mask].ravel()

    # Run all metric
    print("- Start evaluations")
    scores = {}
    for key, (metric, text) in metrics.items():
        result = metric(seg_array.ravel(), gt_array.ravel())
        scores[key] = result
        print(f'{text}: {result}')

    timer += time.time()
    print("- Evaluation took %.1f s" % timer)
    return scores


def main():
    args = parse()
    seg_array, _ = smart_load(args.path_seg, key=args.dataset_seg)
    gt_array, _ = smart_load(args.path_gt, key=args.dataset_gt)
    _ = run_evaluation(gt_array, seg_array, remove_background=True)


if __name__ == "__main__":
    main()
