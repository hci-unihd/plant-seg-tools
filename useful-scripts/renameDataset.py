import glob
import h5py
import os

source_dir = '/home/common/Datasets/Marion/2017-08-02_17.49.34_stPVB003-2-2xDR5v2_F3_nb25_Marvelous/PlantSeg_final/Cells/temp'


for in_file in glob.glob(os.path.join(source_dir, '*.h5')):
    filename = os.path.split(in_file)[1]

    print(f'Renaming {filename}')
    with h5py.File(in_file, 'r') as f:
        f['paintera'] = f['segmentation']
        del f['segmentation']
        f['segmentation'] = f['merged_50000']
        del f['merged_50000']
        f.close()
print('Done!')