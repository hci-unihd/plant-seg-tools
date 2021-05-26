import numpy as np
from scipy.ndimage import zoom
from plantsegtools.utils.io import smart_load, create_h5
from skimage.segmentation import find_boundaries
from plantsegtools.postprocess import LMC_CONFIG_PATH
import yaml
import os
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundaries-path", type=str, help='path to boundaries predictions file')
    parser.add_argument("--nuclei-path", type=str, help='path to nuclei segmentation file')
    parser.add_argument("--seg2pmap", action='store_true', default=False,
                        help='In order to export the segmentation as h5.')
    parser.add_argument("--scaling-pre", default=[1, 1, 1], nargs='+', type=int,
                        help='Scaling factor for the segmentation')
    parser.add_argument("--scaling-post", default=[1, 1, 1], nargs='+', type=int,
                        help='Scaling factor for the segmentation')
    return parser.parse_args()


def load_config():
    with open(LMC_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config


def seg2pmap(nuclei_seg):
    n_boundaries = find_boundaries(nuclei_seg)
    nuclei_mask = nuclei_seg > 0.5
    nuclei_mask[n_boundaries] = 0
    return nuclei_mask.astype(np.float32)


def _prepare_stack_for_lmc():
    args = parse()
    scaling_pre = np.array(args.scaling_pre)

    boundaries, voxel_size = smart_load(args.boundaries_path, key='predictions')
    assert boundaries.ndim in {3, 4}
    boundaries = boundaries[0] if boundaries.ndim == 4 else boundaries

    if np.prod(scaling_pre) != 1:
        boundaries = zoom(boundaries, scaling_pre, order=1)
        voxel_size *= scaling_pre
    boundaries_shape = np.array(boundaries.shape)

    nuclei_pmap, _ = smart_load(args.nuclei_path)
    assert nuclei_pmap.ndim in {3, 4}
    nuclei_pmap = nuclei_pmap[0] if nuclei_pmap.ndim == 4 else nuclei_pmap
    nuclei_pmap_shape = np.array(nuclei_pmap.shape)

    if args.seg2pmap:
        print('transforming nuclei segmentation in pmaps')
        nuclei_pmap = seg2pmap(nuclei_pmap)

    if not np.allclose(nuclei_pmap_shape, boundaries_shape):
        print(f" -fix nuclei predictions shape {nuclei_pmap_shape} to boundary predictions size, {boundaries_shape}")
        nuclei_pmap = zoom(nuclei_pmap, boundaries_shape / nuclei_pmap_shape, order=1)

    boundaries = boundaries[None, ...] if boundaries.ndim == 3 else boundaries
    nuclei_pmap = nuclei_pmap[None, ...] if nuclei_pmap.ndim == 3 else nuclei_pmap

    base, _ = os.path.splitext(args.boundaries_path)
    filename = base.split('/')[-1]
    base_dir, _ = os.path.split(base)

    res_dir = f'{base_dir}/LMC_base/'
    print(f"Preparing all results in {res_dir}")
    os.makedirs(res_dir, exist_ok=True)

    boundaries_path = f'{res_dir}/{filename}_boundaries_predictions.h5'
    create_h5(boundaries_path, boundaries, key='predictions', voxel_size=voxel_size, mode='w')

    nuclei_path = f'{res_dir}/{filename}_nuclei_predictions.h5'
    create_h5(nuclei_path, nuclei_pmap, key='predictions', voxel_size=voxel_size, mode='w')

    config = load_config()
    config['path'] = boundaries_path
    config['segmentation']['nuclei_predictions_path'] = nuclei_path
    out_config = f'{res_dir}/config_lmc.yaml'
    with open(out_config, 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    _prepare_stack_for_lmc()
