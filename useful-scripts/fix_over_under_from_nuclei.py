import numpy as np
from scipy.ndimage import zoom
from plantsegtools.utils.io import smart_load, create_h5, create_tiff
from plantsegtools.postprocess.seg_nuclei_consistency import fix_over_under_segmentation_from_nuclei
import os
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg-path", type=str, help='path to segmentation file')
    parser.add_argument("--nuclei-seg-path", type=str, help='path to nuclei segmentation file')
    parser.add_argument("--t-merge", type=float, default=0.33, help='Overlap merging threshold, between 0-1')
    parser.add_argument("--t-split", type=float, default=0.66, help='Overlap split threshold, between 0-1')
    parser.add_argument("--quantiles", default=[0.3, 0.99], nargs='+', type=float,
                        help='Nuclei size below and above the defined quantiles will be ignored')
    parser.add_argument("--scaling", default=[1, 2, 2], nargs='+', type=int,
                        help='Scaling factor for the segmentation')
    parser.add_argument("--export-h5",
                        action='store_true', default=False,
                        help='In order to export the segmentation as h5.')
    parser.add_argument("--boundaries-path", type=str, help='path to boundaries predictions file')
    return parser.parse_args()


def _fix_over_under_segmentation_from_nuclei():
    args = parse()
    scaling = np.array(args.scaling)

    cell_seg, voxel_size = smart_load(args.seg_path, key='segmentation')
    if np.prod(scaling) != 1:
        cell_seg = cell_seg[::scaling[0], ::scaling[1], ::scaling[2]]
        voxel_size *= scaling
    cell_seg_shape = np.array(cell_seg.shape)

    nuclei_seg, _ = smart_load(args.nuclei_seg_path)
    nuclei_seg_shape = np.array(nuclei_seg.shape)
    if not np.allclose(nuclei_seg_shape, cell_seg_shape):
        print(f" -fix nuclei segmentation shape {nuclei_seg_shape} to cell segmentation size, {cell_seg_shape}")
        nuclei_seg = zoom(nuclei_seg, cell_seg_shape / nuclei_seg_shape, order=0)

    if args.boundaries_path is not None:
        boundaries, _ = smart_load(args.boundaries_path, key='predictions')
        boundaries = boundaries[0] if boundaries.ndim == 4 else boundaries
        boundaries_shape = np.array(boundaries.shape)
        if not np.allclose(boundaries_shape, cell_seg_shape):
            # fix boundary shape if necessary
            print(f" -fix boundaries shape {boundaries_shape} to cell segmentation size, {cell_seg_shape}")
            boundaries = zoom(boundaries, cell_seg_shape / boundaries_shape, order=0)

    else:
        boundaries, _ = None, None

    fix_cell_seg = fix_over_under_segmentation_from_nuclei(cell_seg,
                                                           nuclei_seg,
                                                           threshold_merge=args.t_merge,
                                                           threshold_split=args.t_split,
                                                           quantiles_nuclei=(args.quantiles[0], args.quantiles[1]),
                                                           boundary=boundaries)

    base, ext = os.path.splitext(args.seg_path)
    out_path = f'{base}_nuclei_fixed'
    if args.export_h5:
        out_path += ".h5"
        create_h5(out_path, fix_cell_seg, key='segmentation', voxel_size=voxel_size)
    else:
        out_path += ".tiff"
        create_tiff(out_path, fix_cell_seg, voxel_size=voxel_size)


if __name__ == '__main__':
    _fix_over_under_segmentation_from_nuclei()
