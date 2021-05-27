import numpy as np
from scipy.ndimage import zoom
from plantsegtools.utils.io import smart_load, create_h5
import os
import argparse
from elf.segmentation.watershed import watershed
import vigra
import vigra.filters as ff


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundaries-path", type=str, help='path to boundaries predictions file')
    parser.add_argument("--nuclei-path", type=str, help='path to nuclei segmentation file')
    parser.add_argument("--scaling", default=[1.0, 1.0, 1.0], nargs='+', type=float,
                        help='Scaling factor for the segmentation')
    return parser.parse_args()


def seeded_dt_ws(input_,
                 threshold, seeds, sigma_weights=2., min_size=100, alpha=.9, pixel_pitch=None):

    # threshold the input and compute distance transform
    thresholded = (input_ > threshold).astype('uint32')
    dt = vigra.filters.distanceTransform(thresholded, pixel_pitch=pixel_pitch)

    # normalize and invert distance transform
    dt = 1. - (dt - dt.min()) / dt.max()

    # compute weights from input and distance transform
    if sigma_weights > 0.:
        hmap = alpha * ff.gaussianSmoothing(input_, sigma_weights) + (1. - alpha) * dt
    else:
        hmap = alpha * input_ + (1. - alpha) * dt

    # compute watershed
    ws, max_id = watershed(hmap, seeds, size_filter=min_size)
    return ws, max_id


def _seeded_ws_from_nuclei_seg():
    args = parse()
    scaling_pre = np.array(args.scaling)

    boundaries, voxel_size = smart_load(args.boundaries_path, key='predictions')
    assert boundaries.ndim in {3, 4}
    boundaries = boundaries[0] if boundaries.ndim == 4 else boundaries

    if abs(np.prod(scaling_pre) - 1) > 1e-8:
        print(" -scaling boundary predictions")
        boundaries = zoom(boundaries, scaling_pre, order=1)
        voxel_size *= scaling_pre
    boundaries_shape = np.array(boundaries.shape)

    nuclei_seg, _ = smart_load(args.nuclei_path)
    assert nuclei_seg.ndim in {3, 4}
    nuclei_seg = nuclei_seg[0] if nuclei_seg.ndim == 4 else nuclei_seg
    nuclei_pmap_shape = np.array(nuclei_seg.shape)

    if not np.allclose(nuclei_pmap_shape, boundaries_shape):
        print(f" -fix nuclei predictions shape {nuclei_pmap_shape} to boundary predictions size, {boundaries_shape}")
        nuclei_seg = zoom(nuclei_seg, boundaries_shape / nuclei_pmap_shape, order=0)

    boundaries = boundaries.astype(np.float32)
    boundaries = boundaries / np.max(boundaries)

    nuclei_seg = nuclei_seg.astype('uint32')
    cell_seg, _ = seeded_dt_ws(boundaries, 0.5, nuclei_seg)
    nuclei_seg = nuclei_seg.astype('uint16')
    cell_seg = cell_seg.astype('uint16')

    base, _ = os.path.splitext(args.boundaries_path)
    filename = base.split('/')[-1]
    base_dir, _ = os.path.split(base)

    res_dir = f'{base_dir}/seeded_ws/'
    print(f" -preparing all results in {res_dir}")
    os.makedirs(res_dir, exist_ok=True)
    boundaries = boundaries[None, ...] if boundaries.ndim == 3 else boundaries
    boundaries_path = f'{res_dir}/{filename}_boundaries_predictions.h5'
    create_h5(boundaries_path, boundaries, key='predictions', voxel_size=voxel_size, mode='w')

    nuclei_path = f'{res_dir}/{filename}_nuclei_predictions.h5'
    create_h5(nuclei_path, nuclei_seg, key='segmentation', voxel_size=voxel_size, mode='w')

    cell_path = f'{res_dir}/{filename}_seeded_ws_cell_segmentation.h5'
    create_h5(cell_path, cell_seg, key='segmentation', voxel_size=voxel_size, mode='w')


if __name__ == '__main__':
    _seeded_ws_from_nuclei_seg()
