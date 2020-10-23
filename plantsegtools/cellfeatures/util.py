import h5py
import numpy as np

colors = {0: 'CO', 1: 'C1',
          2: 'C2', 3: 'C3',
          4: 'C4', 5: 'C5',
          6: 'C6', 7: 'C7',
          8: 'C8', 9: 'C9',
          10: 'C0'}


def load_data(h5_path, dataset_path, return_labels=False, remove_zero=True, shuffle=True):
    with h5py.File(h5_path, 'r') as f:
        seg = f['segmentation'][...]
        voxel_res = f['segmentation'].attrs['element_size_um']
        print(f'voxels resolution: {voxel_res}')
        print(f'stack shape: {seg.shape}')
        
    dataset = np.load(dataset_path, allow_pickle=True).item()

    if return_labels:
        labels = np.unique(seg)
        # remove 0 and shuffle
        if remove_zero:
            labels = labels[1:]

        if shuffle:
            np.random.shuffle(labels)
    else:
        labels = None
    return seg, voxel_res, dataset, labels
