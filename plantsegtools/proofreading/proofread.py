from skimage import data
import napari
import h5py
import numpy as np
from skimage.transform import pyramid_gaussian
#from plantsegtools.utils import smart_load
from plantsegtools.utils import create_h5, del_h5_key


class BasicProofread:
    def __init__(self, path, size=1000, stride=None, datasets=None):
        self.path = path
        self.size = size
        self.stride = stride
        self.datasets = datasets

        if datasets is None:
            self.datasets = {'raw': 'raw', 'label': 'label', 'predictions': None}

        self.data, shapes = {}, []
        self.croped_data = {}

        with h5py.File(self.path, 'r') as f:
            for key, value in self.datasets.items():
                if value is not None:
                    self.data[key] = f[value][...]
                    shapes.append(f[value].shape)

        self.shape = shapes[0]
        self.x_pos = (self.shape[1]) // 2
        self.y_pos = (self.shape[2]) // 2
        self.stride = self.size // 4

        print(self.x_pos, self.y_pos)
        self.get_crop()

    def get_crop(self):
        for key, value in self.data.items():
            self.croped_data[key] = value[0,
                                          self.x_pos - self.size//2:self.x_pos + self.size//2,
                                          self.y_pos - self.size//2:self.y_pos + self.size//2]

    def move(self, x, y):
        self.x_pos += x
        self.y_pos += y
        self.x_pos = max(0, min(self.x_pos, self.shape[1] - self.size))
        self.y_pos = max(0, min(self.y_pos, self.shape[2] - self.size))
        self.get_crop()

    def update(self, viewer):
        viewer.layers[0].data = self.croped_data['raw']
        viewer.layers[1].data = self.croped_data['label']

    def __call__(self):
        with napari.gui_qt():
            viewer = napari.Viewer()
            for key, value in self.croped_data.items():
                if key is 'raw' or key is 'predictions':
                    viewer.add_image(value, name=key, multiscale=False)
                elif key is 'label':
                    all_labels = np.unique(value)
                    _color = {0: np.zeros(4), None: np.zeros(4)}
                    for label in all_labels:
                        _color[label] = np.random.rand(4)
                        _color[label][-1] = 1.
                    viewer.add_labels(value, name=key, multiscale=False, color=_color)

                @viewer.bind_key('Control-Left', overwrite=True)
                def move_left(viewer):
                    self.move(0, -self.stride)
                    self.update(viewer)

                @viewer.bind_key('Control-Right', overwrite=True)
                def move_right(viewer):
                    self.move(0, self.stride)
                    self.update(viewer)

                @viewer.bind_key('Control-Up', overwrite=True)
                def move_up(viewer):
                    self.move(-self.stride, 0)
                    self.update(viewer)

                @viewer.bind_key('Control-Down', overwrite=True)
                def move_down(viewer):
                    self.move(self.stride, 0)
                    self.update(viewer)

                @viewer.bind_key('S', overwrite=True)
                def save_current(viewer):
                    del_h5_key(self.path, key='label_edit')
                    create_h5(self.path, self.data['label'], key='label_edit')
                    print('labels saved')

                @viewer.bind_key('H', overwrite=True)
                def save_current(viewer):
                    viewer.layers[1].visible = not viewer.layers[1].visible

                @viewer.bind_key('T', overwrite=True)
                def save_current(viewer):
                    print(viewer.layers[1]._selected_color)
                    print(viewer.layers[1].color_mode)
                    print(viewer.layers[1].selected_label)
                    viewer.layers[1]._selected_color = [0, 0, 0, 0]
                    viewer.layers[1].refresh()
                    viewer.layers[1].events.selected_label()

                @viewer.bind_key('Control-=', overwrite=True)
                def zoom_in(viewer):
                    self.size = int(self.size * 1.25)
                    self.get_crop()
                    self.update(viewer)

                @viewer.bind_key('Control--', overwrite=True)
                def zoom_out(viewer):
                    self.size = int(self.size / 1.25)
                    self.get_crop()
                    self.update(viewer)


if __name__ == '__main__':
    BasicProofread("../../datasets/hypocotyl/train/0_19-0521-21_3.h5")()
