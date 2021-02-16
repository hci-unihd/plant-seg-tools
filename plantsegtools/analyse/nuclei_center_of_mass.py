import argparse
import os
import h5py
import glob
import csv
from skimage import measure
from scipy.ndimage import measurements
import numpy as np


def parse():
    parser = argparse.ArgumentParser(description='Export centers of mass  and fluorescence signal from the nuclei pmaps')
    # Required
    parser.add_argument('--path', type=str, help='Path to the directory containing nuclei pmaps', required=True)
    # Optional
    parser.add_argument('--crop', type=str, help='[ZYX] cropping to apply (e.g. "[:, 0:620, 420:1750]")', default='[:,:,:]', required=False)
    parser.add_argument('--threshold', type=float, help='Probability map threshold', default=0.7, required=False)
    parser.add_argument('--pmaps_dataset', type=str, help='Name of the dataset inside the H5 file', default='predictions', required=False)
    parser.add_argument('--signal', type=str, help='Name of the dataset from which the signal intensity at the nucleus position should be retrieved', default=None, required=False)
    return parser.parse_args()


def writecsv(indict, filepath, signal = None):
    with open(filepath, 'w') as f:
        fieldnames = ['Label', 'com_z', 'com_y', 'com_x', 'signal_mean', 'signal_median']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if signal is None:
            data = [dict(zip(fieldnames, [k, f'{v[0]:.3f}', f'{v[1]:.3f}', f'{v[2]:.3f}', f'NA', f'NA'])) for k, v in indict.items()]
        else:
            data = [dict(zip(fieldnames, [k, f'{v[0]:.3f}', f'{v[1]:.3f}', f'{v[2]:.3f}', f'{v[3]:.3f}', f'{v[4]:.3f}'])) for k, v in indict.items()]
        writer.writerows(data)


# TODO  parallelize over timepoints!
# TODO  extract median fluo intensity in the nucleus
def main():
    args = parse()
    # Setup input path
    if os.path.isfile(args.path):
        all_files = [args.path]
    elif os.path.isdir(args.path):
        all_files = glob.glob(os.path.join(args.path, f'*{H5_FORMATS}'))
    else:
        raise NotImplementedError

    for i, in_file in enumerate(all_files, 1):
        # Prep IO
        path = os.path.dirname(in_file)
        fname = os.path.splitext(os.path.basename(in_file))[0]
        outfname = f"nuclei_{fname}.csv"
        outpath = os.path.join(path, outfname)

        # Main  loop
        print(f"Processing {fname}.h5 ({i}/{len(all_files)})")
        with h5py.File(in_file, 'r') as f:
            print("Computing center of mass...", end='')
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
            print(f"done, {len(labels)} labels processed.")
            # Compute signal at each nuclei masks in data
            if args.signal is not None:
                print("Computing signal...", end='')
                for label in labels:
                    num_of_voxels = (connected_components == label).astype(np.uint8).sum()
                    avg_intensity = np.sum(f[args.signal][connected_components == label]) / num_of_voxels  # FIXME  raise 'TypeError: Boolean indexing array has incompatible shape'
                    med_intensity = np.median(f[args.signal][connected_components == label])
                    nuclei_com_dict[label] = nuclei_com_dict[label] + (avg_intensity, med_intensity)
                print('done!')
            else:
                for label in labels:
                    nuclei_com_dict[label] = nuclei_com_dict[label] + (None, None)

            # Writing output
            print(f'Writing to {outfname}')
            writecsv(nuclei_com_dict, outpath, args.signal)
    i += 1


if __name__ == '__main__':
    print("Starting nuclei analyse...")
    main()
    print("Done!")

'''
            for label in nuclei_labels:
              num_of_voxels = (nuclei_segmentation == label).astype(np.uint8).sum()
              avg_intensity = np.sum(raw[nuclei_segmentation == label]) / num_of_voxels
              print(f'Nuclei label: {label}, avg intensity: {avg_intensity}')
            '''
