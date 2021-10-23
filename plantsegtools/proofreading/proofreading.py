from plantsegtools.proofreading import BasicProofread
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--path-raw', type=str, required=True, help='path to raw file')
    parser.add_argument('-dr', '--dataset-raw', type=str, default='raw',
                        required=False, help='if h5 contains dataset name to raw')
    parser.add_argument('-s', '--path-seg', type=str, required=False, help='path to segmentation file')
    parser.add_argument('-ds', '--dataset-seg', type=str, default='label',
                        required=False, help='if h5 contains dataset name to segmentation')
    parser.add_argument('-xy', '--xy-size', type=int, default=1000, help='field of view size on the xy-plane')
    parser.add_argument('-z', '--z-size', type=int, default=2, help='field of view size on z')
    return parser.parse_args()


def main():
    args = parse()
    path_raw = args.path_raw
    path_seg = args.path_seg

    dataset_raw = args.dataset_raw
    dataset_seg = args.dataset_seg

    xy_size = args.xy_size
    z_size = args.z_size

    BasicProofread(path_raw=path_raw,
                   path_label=path_seg,
                   datasets=(dataset_raw, dataset_seg),
                   xy_size=xy_size,
                   z_size=z_size)()

