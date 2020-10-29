from plantsegtools.utils.io import load_h5, load_tiff, create_h5
from plantsegtools.utils.edit import crop_image
from skimage.segmentation import find_boundaries
import tifffile
import numpy as np
import csv
import os
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help='path to segmentation file')
    parser.add_argument("--path-csv", type=str, help='path to csv file')
    parser.add_argument("--is-tiff", action='store_true', help='This flag indicates that the segmentation file is tiff.')
    parser.add_argument("--is-h5", action='store_true', help='This flag indicates that the segmentation file is h5.')
    parser.add_argument("--h5-dataset", default='segmentation', help='h5 internal dataset name')
    parser.add_argument("--crop", default=[0, 0, 0, -1, -1, -1], nargs='+', type=int, help='crop the dataset, takes as input a bounding box. eg --crop 10, 0, 0 15, -1, -1.')
    parser.add_argument("--seed", default=0, type=int, help='change seed for different random rgb')
    return parser.parse_args()


def parse_csv(csv_path):
    with open(csv_path, 'r') as f:
        spamreader = csv.reader(f, delimiter=',')
        spamreader = list(spamreader)[1:]
        seg_labels, cell_types = [], []
        for row in spamreader:
            seg_labels.append(int(row[0]))
            cell_types.append(int(row[1]))
            
        return seg_labels, cell_types


def load_tiff_crop(path, crop):
    seg, voxel_resolution = load_tiff(path)
    seg = crop(seg, start=crop[:3], end=crop[3:])
    return seg


def mask_boundaries(seg):
    mask = find_boundaries(seg)
    seg[mask] = 0
    return seg


def map_celltype2rgbimage(cell_types, seg_labels, seg, seed):
    
    np.random.seed(seed)
    random_color = np.random.randint(255, size=(np.max(cell_types) + 1, 3))
    random_color[0] = 0
    
    rgb_image = np.zeros((seg.shape[0], seg.shape[1], seg.shape[2], 3), dtype=np.uint8)
    for label, cell_type in zip(seg_labels, cell_types):
        _mask = seg == label
        rgb_image[_mask, :] = random_color[cell_type]
    return rgb_image


def celltype2rgb():
    args = parse()
    
    path = args.path
    is_tiff = args.is_tiff
    is_h5 = args.is_h5
    cell_type_csv = args.path_csv
    h5_dataset = args.h5_dataset
    seed = args.seed
    crop = args.crop

    if is_tiff:
        seg, voxel_size = load_tiff(path)
        seg = crop_image(seg, start=crop[:3], end=crop[3:])
    elif is_h5:
        seg, voxel_size = load_h5(path, h5_dataset, slices=(slice(crop[0], crop[3]),
                                                            slice(crop[1], crop[4]),
                                                            slice(crop[2], crop[5])))
    else:
        print('Segemntation file should be either tiff or h5, please specify which one by using the flag is_tiff or is_h5')
        raise NotImplementedError
    
    seg = mask_boundaries(seg)
    seg_labels, cell_types = parse_csv(cell_type_csv)
    
    rgb_image = map_celltype2rgbimage(cell_types, seg_labels, seg, seed)
    base, ext = os.path.splitext(path)
    out_path = f'{base}_celltype_rgb{ext}'
    if is_tiff:
        tifffile.imwrite(out_path, rgb_image)
    else:
        create_h5(out_path, rgb_image, voxel_size=voxel_size)


if __name__ == '__main__':
    celltype2rgb()
