import argparse
import os
import h5py
import glob
import csv
import time
from datetime import datetime
import numpy as np
from skimage import measure
from scipy.ndimage import measurements
from plantsegtools.utils import H5_FORMATS

try:
    import ray
except ImportWarning:
    pass


def parse():
    parser = argparse.ArgumentParser(description='Process  the nuclei pmaps to compute the center of mass of each nucleus, retrieve fluorescence signal and map to cell segmentation')
    # Required
    parser.add_argument('--path', type=str, help='Path to the directory containing nuclei pmaps', required=True)
    # Optional
    parser.add_argument('--crop', type=str, help='[ZYX] cropping to apply (e.g. "[:, 0:620, 420:1750]")', default='[:,:,:]', required=False)
    parser.add_argument('--threshold', type=float, help='Probability map threshold', default=0.7, required=False)
    parser.add_argument('--pmaps_dataset', type=str, help='Name of the dataset inside the H5 file', default='predictions', required=False)
    parser.add_argument('--signal', type=str, help='Name of the dataset from which the signal intensity at the nucleus position should be retrieved', default=None, required=False)
    parser.add_argument('--cell_segmentation', type=str, help='Name of the dataset containing the cell segmentation to use for mapping nuclei to cell', default=None, required=False)
    parser.add_argument('--multiprocessing', help='Define the number of cores to use for parallel processing.'
                                                  ' Default value (-1) will try to parallelize over'
                                                  ' all available processors.', default=-1, type=int)
    return parser.parse_args()


def writecsv(indict, filepath, signal=None, cell_segmentation=None):
    with open(filepath, 'w') as f:
        fieldnames = ['nuclei_label', 'com_z', 'com_y', 'com_x', 'signal_mean', 'signal_median', 'cell_label']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if (signal, cell_segmentation) == (None, None):
            data = [dict(zip(fieldnames, [k, f'{v[0]:.0f}', f'{v[1]:.0f}', f'{v[2]:.0f}', f'NA', f'NA', f'NA'])) for k, v in indict.items()]
        elif cell_segmentation is None:
            data = [dict(zip(fieldnames, [k, f'{v[0]:.0f}', f'{v[1]:.0f}', f'{v[2]:.0f}', f'{v[3]:.3f}', f'{v[4]:.3f}', f'NA'])) for k, v in indict.items()]
        elif signal is None:
            data = [dict(zip(fieldnames, [k, f'{v[0]:.0f}', f'{v[1]:.0f}', f'{v[2]:.0f}', f'NA', f'NA', f'{v[5]}'])) for k, v in indict.items()]
        else:
            data = [dict(zip(fieldnames, [k, f'{v[0]:.0f}', f'{v[1]:.0f}', f'{v[2]:.0f}', f'{v[3]:.3f}', f'{v[4]:.3f}', f'{v[5]}'])) for k, v in indict.items()]
        writer.writerows(data)


def avg_median(signal, connected_components, label):
    avg_intensity = np.mean(f[signal][connected_components == label])
    med_intensity = np.median(f[signal][connected_components == label])
    return avg_intensity, med_intensity

if __name__ == '__main__':
    args = parse()
    print(f"[{datetime.now().strftime('%d-%m-%y %H:%M:%S')}] start pipeline setup, parameters: {vars(args)}")

    multiprocessing = args.multiprocessing

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
        print(f"[{datetime.now().strftime('%d-%m-%y %H:%M:%S')}]"
              f" start processing file: {fname}.h5  ({i}/{len(all_files)})")
        timer = time.time()
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
            nuclei_com =  measurements.center_of_mass(nuclei_mask, connected_components, labels)
            nuclei_com_dict = {}
            for label, coord in zip(labels, nuclei_com):
                nuclei_com_dict[label] = coord
            print(f"done: {len(labels)} labels processed.")
            # Compute signal at each nuclei masks in data
            if args.signal is not None: # TODO  parallelize! either over  each nuclei labels
                print("Computing signal...", end='')
                if multiprocessing < 1:
                    ray.init()
                else:
                    ray.init(num_cpus=multiprocessing)
                connected_components_id = ray.put(connected_components)

                @ray.remote
                def remote_avg_median(signal, _connected_components, label, nuclei_com_dict):
                    avg, med = avg_median(signal,  _connected_components, label)
                    nuclei_com_dict[label] = nuclei_com_dict[label] + (avg, med)
                    return nuclei_com_dict[label]

                tasks = [remote_avg_median.remote(args.signal, connected_components_id, label, nuclei_com_dict) for label in labels]
                results = ray.get(tasks)
                ray.shutdown()
                print(results)
                #for label in labels:
                #    avg_intensity, med_intensity = avg_median(args.signal, connected_components, label)
                #    nuclei_com_dict[label] = nuclei_com_dict[label] + (avg_intensity, med_intensity)
                print('done!')
            else:
                for label in labels:
                    nuclei_com_dict[label] = nuclei_com_dict[label] + (None, None)
            if args.cell_segmentation is not None:
                print("Mapping nuclei to cells...", end='')
                for label in labels:
                    cell_label =  f[args.cell_segmentation][nuclei_com_dict[label][0], nuclei_com_dict[label][1], nuclei_com_dict[label][2]]
                    nuclei_com_dict[label] = nuclei_com_dict[label] + (cell_label,)
                print('done!')
            else:
                for label in labels:
                    nuclei_com_dict[label] = nuclei_com_dict[label] + (None,)
            # Writing output
            print(f'Writing to {outfname}')
            writecsv(nuclei_com_dict, outpath, args.signal, args.cell_segmentation)
    print(f"[{datetime.now().strftime('%d-%m-%y %H:%M:%S')}]"
        f" process complete in {time.time() - timer: .2f}s,")
    i += 1