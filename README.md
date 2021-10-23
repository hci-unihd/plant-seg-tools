# plant-seg-tools
Simple python tools for plant-seg https://github.com/hci-unihd/plant-seg

## Install tools
* Install `plant-seg` from https://github.com/hci-unihd/plant-seg
* activate `plant-seg` environment using:
```
conda activate plant-seg
```
* clone this repository locally on our machine, in the terminal navigate to the desired install location and execute:
```
git clone https://github.com/hci-unihd/plant-seg-tools.git
```
* install `plant-seg-tools`:
```
cd plant-seg-tools
pip install .
```

## Additional dependencies
While having your conda environment activated:
* In oder to use `seg2mesh`
```
conda install -c conda-forge ray vtk
```
* In order to use the `trimesh` backend of `plantsegtools/meshes`, you will need to install `trimesh` using:
```
conda install -c conda-forge trimesh
```
* In order to use the training configurator wizard you need to install `PyInquirer`
```
pip install PyInquirer
```

## Install proofreading tool
* create a new conda environment (at the moment plant-seg and the proofreading tools are not compatible). 
On a fresh terminal execute:
```
conda create -n plant-seg-proofreading -c conda-forge python numpy numba scipy matplotlib scikit-image pyaml h5py tqdm napari pyqt=5.12.3
conda activate plant-seg-proofreading
```
* clone this repository locally on our machine, in the terminal navigate to the desired install location and execute:
```
git clone https://github.com/hci-unihd/plant-seg-tools.git
```
* install `plant-seg-tools`:
```
cd plant-seg-tools
pip install .
```

## Index
* [Segmentation to Meshes](##Segmentation to Meshes)
* [Proofreading tool](#Proofreading tool)
* [Automated Segmentation Proofread from seeds](#Automated Segmentation Proofread from seeds)

## Segmentation to Meshes

### Basic usage
From the project root (`plant-seg-tools`) run the `seg2mesh` script using:
```
python useful-scripts/seg2mesh.py --path 'PATHTOSEGMENTATIONSTACK'
```

#### Optional arguments guide
* `--new-base`: optional custom saving directory. 
If not given the ply will be saved in the same dir as the source.
* `--h5-dataset`: h5 internal dataset name. Default: segmentation.
* `--labels`: List of labels to process. By default, the script will process all labels.
* `--step-size`: Step size for the marching cube algorithm, larger steps yield a coarser but faster result. Default 2.
* `--crop`: Crop the dataset, takes as input a bounding box. eg --crop 10, 0, 0 15, -1, -1.
* `--voxel-size`: Voxel size [Z, Y, X] of the segmentation stack. By default, voxel size is read from the source file, if
this is not possible voxel-size is set to [1, 1, 1].
* `--min-size`: Minimum cell size. Default 50.
* `--max-size`: Maximum cell size. Default inf.
* `--relabel`:  If this argument is passed the pipeline will relabel the segmentation.
This will ensure the contiguity of each segment but will change the labels.
* `--check-cc`: If this argument is passed the pipeline will check if each label is has a single connected component (cc).
If multiple cc are present only the largest will be processed.
* `--ignore-labels`: List of labels to ignore. By default, only background (label 0) is ignored.
* `--reduction`: If reduction > 0 a decimation filter is applied.MaxValue: 1.0 (100%reduction).
* `--smoothing`: To apply a Laplacian smoothing filter.
* `--use-ray`: If you use ray flag is used the multiprocessing flag is managed by ray.
* `--multiprocessing`: Define the number of cores to use for parallel processing. 
Default value (-1) will try to parallelize over all available processors.

## Proofreading tool

### Basic usage
From the project root (`plant-seg-tools`) run the proofreading tool using:
* if the stack is coming from PlantSeg
```
python useful-scripts/proofreading.py --path-raw 'PATHTORAWSTACK' --path-seg 'PATHTOSEGMENTATIONSTACK' --dataset-seg segmentation
```
or in short
```
python useful-scripts/proofreading.py --r 'PATHTORAWSTACK' -s 'PATHTOSEGMENTATIONSTACK' -ds segmentation
```
* if you want to continue working on a stack exported with the proofreading tool 
(or if segmentation and raw images are together in the same h5 file):
```
python useful-scripts/proofreading.py -r 'PATHTORAWSTACK (convetional name *_proofreading.h5')'
```
#### Keybinding Guide:
* `s`: Save stack
* `n`: Merge or split from seeds
* `ctrl+n`: Undo merge or split from seeds
* `c`: Clean seeds
* `o`: Mark/un-mark correct segmentation
* `b`: show/un-show correct segmentation layer
* `j`: Update boundaries from segmentation
* `k`: Update segmentation from boundaries  
#### Navigation
* `ctrl + arrows`: to move the field of view
* `alt + down/up arrows`: to increase or decrease the field of view

#### Optional arguments guide
* `-dr` or `--dataset-raw`: name of the dataset containing the raw boundary image.
Only used if raw boundary image is in `h5` format. default `raw`.
* `-ds` or `--dataset-seg`: name of the dataset containing the cell segmentation.
Only used if segmentation is in `h5` format. default `label`.
* `-xy` or `--xy-size`: field of view size along the xy-plane. Larger field of view might slow down the tool.
* `-z` or `--z-size`: field of view size along z. Only for 3D. Larger field of view might slow down the tool.



## Automated Segmentation Proofread from seeds

### Basic usage
```
python useful-scripts/fix_over_under_from_nuclei.py --seg-path 'PATHTOSEGMENTATIONSTACK' --nuclei-seg-path 'PATHTONUCLEISEGMENTATIONSTACK' 
```

#### Optional arguments guide
  * `--t-merge`: Overlap merging threshold, between 0-1.
  * `--t-split`: Overlap split threshold, between 0-1.
  * `--quantiles`: Nuclei size below and above the defined quantiles will be ignored.
  * `--scaling`: Scaling factor for the segmentation.
  * `--export-h5`:  In order to export the segmentation as h5.
  * `--boundaries-path`: path to boundaries predictions file.


