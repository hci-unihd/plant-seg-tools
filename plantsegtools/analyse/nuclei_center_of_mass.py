import argparse
import os
import h5py
import glob
import csv
from skimage import measure
from scipy.ndimage import measurements
import numpy as np


# TODO  parallelize over timepoints!
# TODO  extract median fluo intensity in the nucleus
def main():
    parser = argparse.ArgumentParser(description='Export centers of mass from the nuclei pmaps')
    parser.add_argument('--in_dir', type=str, help='Path to the directory containing nuclei pmaps', required=True)
    parser.add_argument('--threshold', type=float, help='Probability map threshold', default=0.7, required=False)
    parser.add_argument('--pmaps_dataset', type=str, help='Name of the dataset inside the H5 file', default='predictions', required=False)
    args = parser.parse_args()

    for in_file in glob.glob(os.path.join(args.in_dir, '*.h5')):
        print(f'Processing {in_file}')
        filename = os.path.splitext(os.path.basename(in_file))[0]
        with h5py.File(in_file, 'r') as f:
            pmaps = f[args.pmaps_dataset][...]
            if pmaps.ndim == 4:
                # take the first channel from the CZYX pmaps array
                pmaps = pmaps[0]

            nuclei_mask = (pmaps > args.threshold).astype(np.uint8)
            connected_components = measure.label(nuclei_mask)

            # skip 0-label
            labels = np.unique(connected_components)[1:]
            # compute centers of mass
            nuclei_com = measurements.center_of_mass(nuclei_mask, connected_components, labels)
            nuclei_com_dict = {}
            for label, coord in zip(labels, nuclei_com):
                nuclei_com_dict[label] = coord
            outfilename = 'com_nuc_' + filename + '.csv'
            outfilepath = os.path.join(args.in_dir, outfilename)
            print(f'Writing to {outfilename}')
            writecsv(nuclei_com_dict, outfilepath)


def writecsv(indict, filepath):
    with open(filepath, 'w') as f:
        fieldnames = ['Label', 'com_z', 'com_y', 'com_x']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        data = [dict(zip(fieldnames, [k, f'{v[0]:.3f}', f'{v[1]:.3f}', f'{v[2]:.3f}'])) for k, v in indict.items()]
        writer.writerows(data)


if __name__ == '__main__':
    main()
    print("Done!")

'''

            for label in nuclei_labels:
              num_of_voxels = (nuclei_segmentation == label).astype(np.uint8).sum()
              avg_intensity = np.sum(raw[nuclei_segmentation == label]) / num_of_voxels
              print(f'Nuclei label: {label}, avg intensity: {avg_intensity}')
            '''