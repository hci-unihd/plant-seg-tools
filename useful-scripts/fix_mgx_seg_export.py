from plantsegtools.utils.io import smart_load, create_h5, create_tiff
import os
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg-path", type=str, help='path to segmentation file')
    parser.add_argument("--seg-dataset", type=str, default='segmentation', help='internal h5 dataset')
    return parser.parse_args()


def _fix_mgx_seg_export():
    args = parse()
    segmentation, voxel_size = smart_load(args.seg_path, key=args.seg_dataset)
    segmentation = segmentation[:, ::-1]

    out_path = os.path.splitext(args.seg_path)[0]
    if args.export_h5:
        out_path += ".h5"
        create_h5(out_path, segmentation, key=args.seg_dataset, voxel_size=voxel_size)
    else:
        out_path += ".tiff"
        create_tiff(out_path, segmentation, voxel_size=voxel_size)


if __name__ == '__main__':
    _fix_mgx_seg_export()
