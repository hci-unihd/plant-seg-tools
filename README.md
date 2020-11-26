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

## additional dependencies
While having your conda environment activated:
* In order to use the `vtk` backend of `plantsegtools/meshes`, you will need to install `vtk` using:
```
conda install -c conda-forge vtk
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
conda create -n plant-seg-proofreading -c conda-forge python=3.7 numpy numba scipy matplotlib scikit-image pyaml h5py tqdm napari
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

## Usage
From the `plant-seg-tools` directory run the proofreading tool using:
* if the stack is coming from PlantSeg
```
python useful-scripts/proofreading.py --path-raw 'PATHTORAWSTACK' --path-seg 'PATHTOSEGMENTATIONSTACK' --dataset-seg segmentation
```
* if you want to continue working on a stack exported with the proofreading tool:
```
python useful-scripts/proofreading.py --path-raw 'PATHTORAWSTACK (convetional name *_proofreading.h5')'
```
**TODO** document keybinding