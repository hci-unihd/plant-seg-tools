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
* from the `plant-seg-tools` directory run the proofreading tool using:
```
python useful-scripts/proofreading.py --path-raw 'PATHTORAWSTACK' --path-seg 'PATHTOSEGMENTATIONSTACK'
```
This command works for stacks segmented with `plant-seg`

**TODO** documents other options