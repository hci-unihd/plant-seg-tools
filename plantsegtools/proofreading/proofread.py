import os

import napari
import numpy as np
from skimage.measure import label
from skimage.segmentation import find_boundaries, watershed

from plantsegtools.utils import create_h5, smart_load

raw_key = 'raw'
segmentation_key = 'segmentation'
seg_boundaries_key = 'seg-boundaries'


class BasicProofread2D:
    def __init__(self,
                 path_raw,
                 path_label=None,
                 datasets=('raw', 'label'),
                 size=1000,
                 stride=None):

        path_label = path_raw if path_label is None else path_label

        self.datasets = {raw_key: (path_raw, datasets[0]),
                         segmentation_key: (path_label, datasets[1]),
                         }

        self.size = size
        self.stride = stride

        self.data, shapes = {}, []
        for key, values in self.datasets.items():
            _path, _key = values
            stack, voxel_size = smart_load(_path, _key)
            self.data[key] = stack
            shapes.append(stack.shape)

        self.datasets[seg_boundaries_key] = (None, seg_boundaries_key)
        self.data[seg_boundaries_key] = self.get_seg_boundary()

        assert shapes[0] == shapes[1] == shapes[2]

        self.cropped_data = {}
        self.shape = shapes[0]
        self.x_pos = (self.shape[1]) // 2
        self.y_pos = (self.shape[2]) // 2
        self.stride = self.size // 4

        self.get_crop()

    def get_seg_boundary(self):
        segmentation = self.data[segmentation_key]
        return find_boundaries(segmentation)

    def update_boundary(self):
        _local_boundary = find_boundaries(self.cropped_data[segmentation_key])
        self.data[seg_boundaries_key][0, self.x_pos - self.size//2:self.x_pos + self.size//2,
                                         self.y_pos - self.size//2:self.y_pos + self.size//2] = _local_boundary

    def update_segmentation(self, x, y):
        label_idx = self.cropped_data[segmentation_key][int(x), int(y)]
        mask = self.data[segmentation_key][0] == label_idx
        coords = np.nonzero(mask)
        xmin, xmax = coords[0].min() - 2, coords[0].max() + 2
        ymin, ymax = coords[1].min() - 2, coords[1].max() + 2

        _boundaries = self.data[seg_boundaries_key][0, xmin:xmax, ymin:ymax]
        _mask = self.data[segmentation_key][0, xmin:xmax, ymin:ymax] == label_idx
        seeds = label(_boundaries, background=1)
        seeds[~_mask] = 0
        _seg = watershed(np.ones_like(seeds), markers=seeds)
        _seg += self.data[segmentation_key].max() + 1
        self.data[segmentation_key][0, xmin:xmax, ymin:ymax][_mask] = _seg[_mask].ravel()

    def get_crop(self):
        for key, value in self.data.items():
            self.cropped_data[key] = value[0,
                                           self.x_pos - self.size//2:self.x_pos + self.size//2,
                                           self.y_pos - self.size//2:self.y_pos + self.size//2]

    def move(self, x, y):
        self.x_pos += x
        self.y_pos += y
        self.x_pos = max(0, min(self.x_pos, self.shape[1] - self.size))
        self.y_pos = max(0, min(self.y_pos, self.shape[2] - self.size))
        self.get_crop()

    def update(self, viewer):
        for _layer_key in viewer.layers:
            viewer.layers[_layer_key.name].data = self.cropped_data[_layer_key.name]

    def init_layers(self, viewer):
        viewer.add_image(self.cropped_data[raw_key], name=raw_key, multiscale=False)
        viewer.add_labels(self.cropped_data[segmentation_key], name=segmentation_key, multiscale=False)
        viewer.add_labels(self.cropped_data[seg_boundaries_key], name=seg_boundaries_key, multiscale=False)

    def __call__(self):
        with napari.gui_qt():
            viewer = napari.Viewer()
            self.init_layers(viewer)

            @viewer.bind_key('Control-Left')
            def move_left(viewer):
                """move field of view left"""
                self.move(0, -self.stride)
                self.update(viewer)

            @viewer.bind_key('Control-Right')
            def move_right(viewer):
                """move field of view right"""
                self.move(0, self.stride)
                self.update(viewer)

            @viewer.bind_key('Control-Up')
            def move_up(viewer):
                """move field of view up"""
                self.move(-self.stride, 0)
                self.update(viewer)

            @viewer.bind_key('Control-Down')
            def move_down(viewer):
                """move field of view down"""
                self.move(self.stride, 0)
                self.update(viewer)

            @viewer.bind_key('S')
            def save_current(viewer):
                """create a training stack"""
                seg_path = self.datasets[segmentation_key][0]
                base, ext = os.path.splitext(seg_path)
                seg_path = f'{base}_edit{ext}'
                create_h5(seg_path, self.data[segmentation_key], key='label', mode='w')
                create_h5(seg_path, self.data[raw_key], key=raw_key)
                print('labels saved')

            @viewer.bind_key('J')
            def _update_boundaries(viewer):
                """Update boundaries"""
                self.update_boundary()
                self.get_crop()
                self.update(viewer)

            @viewer.bind_key('K')
            def _update_segmentation(viewer):
                """Update Segmentation under cursor"""
                x, y = viewer.layers[segmentation_key].coordinates
                self.update_segmentation(x, y)
                self.update_boundary()
                self.get_crop()
                self.update(viewer)

            @viewer.bind_key('Control-=')
            def zoom_in(viewer):
                """zoom in"""
                self.size = int(self.size * 1.25)
                self.get_crop()
                self.update(viewer)

            @viewer.bind_key('Control--')
            def zoom_out(viewer):
                """zoom out"""
                self.size = int(self.size / 1.25)
                self.get_crop()
                self.update(viewer)


if __name__ == '__main__':
    BasicProofread2D(path_raw="/home/lcerrone/datasets/hypocotyl/train/0_19-0521-21_3.h5")()
