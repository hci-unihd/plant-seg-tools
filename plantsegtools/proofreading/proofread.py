import os

import napari
import numpy as np
from skimage.measure import label
from skimage.segmentation import find_boundaries, watershed
from skimage.filters import sobel, gaussian
from plantsegtools.utils import create_h5, smart_load

raw_key = 'raw'
segmentation_key = 'segmentation'
seg_boundaries_key = 'seg-boundaries'
seeds_merge_key = 'seeds-merge'
seeds_split_key = 'seeds-split'
seg_correct_key = 'seg-correct'

seg_boundaries_cmap = {0: None,
                       1: (0.05, 0.05, 0.05, 1.)
                       }
seg_correct_cmap = {0: None,
                    1: (0.76388469, 0.02003777, 0.61156412, 1.)
                    }


class BasicProofread:
    def __init__(self,
                 path_raw,
                 path_label=None,
                 datasets=('raw', 'label'),
                 xy_size=1000,
                 z_size=2,
                 stride=None):

        path_label = path_raw if path_label is None else path_label

        self.datasets = {raw_key: (path_raw, datasets[0]),
                         segmentation_key: (path_label, datasets[1]),
                         }

        self.data, shapes = {}, []
        for key, values in self.datasets.items():
            _path, _key = values
            stack, voxel_size = smart_load(_path, _key)
            self.data[key] = stack
            shapes.append(stack.shape)

        self.datasets[seg_boundaries_key] = (None, seg_boundaries_key)
        self.datasets[seeds_merge_key] = (None, seeds_merge_key)
        self.datasets[seeds_split_key] = (None, seeds_split_key)
        self.datasets[seg_correct_key] = (None, seg_correct_key)

        self.data[seg_boundaries_key] = self.get_seg_boundary()
        self.data[seeds_merge_key] = np.empty((0, 3))
        self.data[seeds_split_key] = np.zeros(shapes[0])
        self.data[seg_correct_key] = np.zeros(shapes[0])

        assert shapes[0] == shapes[1]

        self.xy_size = min(xy_size, min(shapes[0][1], shapes[0][2]))
        self.z_size = min(z_size, shapes[0][0])

        self.cropped_data = {}
        self.shape = shapes[0]
        self.z_pos = (self.shape[0]) // 2
        self.x_pos = (self.shape[1]) // 2
        self.y_pos = (self.shape[2]) // 2
        self.stride = self.xy_size // 4 if stride is None else stride

        self.get_crop()

        self.last_slices, self.last_seg = None, None

    def get_slices(self):
        zmin, zmax = max(self.z_pos - self.z_size // 2, 0), min(self.z_pos + self.z_size // 2, self.shape[0])
        zmax = zmax if zmax - zmin > 0 else 1
        xmin, xmax = max(self.x_pos - self.xy_size // 2, 0), min(self.x_pos + self.xy_size // 2, self.shape[1])
        ymin, ymax = max(self.y_pos - self.xy_size // 2, 0), min(self.y_pos + self.xy_size // 2, self.shape[2])
        return slice(zmin, zmax), slice(xmin, xmax), slice(ymin, ymax)

    def get_crop(self):
        _slices = self.get_slices()
        print(f"current slice: {_slices}")
        for key, value in self.data.items():
            if key not in [seeds_merge_key]:
                self.cropped_data[key] = value[_slices]

    def update(self, viewer):
        for _layer_key in viewer.layers:
            if _layer_key.name not in [seeds_merge_key]:
                viewer.layers[_layer_key.name].data = self.cropped_data[_layer_key.name]

    def crop_update(self, viewer):
        self.get_crop()
        self.update(viewer)

    def init_layers(self, viewer):
        viewer.add_image(self.cropped_data[raw_key], name=raw_key, multiscale=False)
        viewer.add_labels(self.cropped_data[segmentation_key], name=segmentation_key, multiscale=False)
        _rmap = np.random.rand(4)
        _rmap[-1] = 1
        print(_rmap)
        viewer.add_labels(self.cropped_data[seg_correct_key],
                          name=seg_correct_key,
                          color=seg_correct_cmap,
                          opacity=1,
                          multiscale=False)
        viewer.add_labels(self.cropped_data[seeds_split_key], name=seeds_split_key)
        viewer.add_points(self.data[seeds_merge_key], name=seeds_merge_key)
        viewer.add_labels(self.cropped_data[seg_boundaries_key],
                          name=seg_boundaries_key,
                          color=seg_boundaries_cmap,
                          opacity=1,
                          multiscale=False)

    def move(self, x, y):
        self.x_pos += x
        self.y_pos += y
        self.x_pos = max(0, min(self.x_pos, self.shape[1] - self.xy_size))
        self.y_pos = max(0, min(self.y_pos, self.shape[2] - self.xy_size))
        self.get_crop()

    def get_seg_boundary(self):
        segmentation = self.data[segmentation_key]
        return find_boundaries(segmentation)

    def update_boundary(self):
        _local_boundary = find_boundaries(self.cropped_data[segmentation_key])
        self.data[seg_boundaries_key][self.get_slices()] = _local_boundary

    @staticmethod
    def _mask_bbox(mask):
        max_shape = mask.shape
        coords = np.nonzero(mask)
        zmin, zmax = max(coords[0].min() - 2, 0), min(coords[0].max() + 2, max_shape[0])
        zmax = zmax if zmax - zmin > 0 else 1
        xmin, xmax = max(coords[1].min() - 2, 0), min(coords[1].max() + 2, max_shape[1])
        ymin, ymax = max(coords[2].min() - 2, 0), min(coords[2].max() + 2, max_shape[2])
        return (slice(zmin, zmax), slice(xmin, xmax), slice(ymin, ymax)), zmin, xmin, ymin

    def update_segmentation(self, z, x, y):
        # create bbox from mask
        label_idx = self.cropped_data[segmentation_key][int(z), int(x), int(y)]
        mask = self.data[segmentation_key] == label_idx
        bbox_slices, _, _, _ = self._mask_bbox(mask)

        _boundaries = self.data[seg_boundaries_key][bbox_slices]
        _mask = self.data[segmentation_key][bbox_slices] == label_idx
        seeds = label(_boundaries, background=1)
        seeds[~_mask] = 0

        _seg = watershed(np.ones_like(seeds), markers=seeds)
        _seg += self.data[segmentation_key].max() + 1
        self.data[segmentation_key][bbox_slices][_mask] = _seg[_mask].ravel()

    def merge_from_seeds(self, points):
        self._save_old(self.get_slices())
        new_label_idx = self.cropped_data[segmentation_key][int(points[0][0]),
                                                            int(points[0][1]),
                                                            int(points[0][2])]

        for i, point in enumerate(points):
            label_idx = self.cropped_data[segmentation_key][int(point[0]),
                                                            int(point[1]),
                                                            int(point[2])]
            mask = self.data[segmentation_key] == label_idx
            self.data[segmentation_key][mask] = new_label_idx

    def split_from_seeds(self, seeds):

        # find seeds location ad label value
        sz, sx, sy = np.nonzero(seeds)
        all_idx = self.cropped_data[segmentation_key][sz, sx, sy]
        all_idx = np.unique(all_idx)

        # create bbox from mask
        mask = np.logical_or.reduce([self.data[segmentation_key] == label_idx for label_idx in all_idx])
        bbox_slices, zmin, xmin, ymin = self._mask_bbox(mask)
        self._save_old(bbox_slices)

        # load main bbox data
        _mask = np.logical_or.reduce([self.data[segmentation_key][bbox_slices] == label_idx for label_idx in all_idx])
        _raw = self.data[raw_key][bbox_slices]

        # create bbox seeds
        _seeds = np.zeros_like(_mask).astype(np.int)
        local_sz = sz + self.z_pos - self.z_size//2 - zmin
        local_sx = sx + self.x_pos - self.xy_size//2 - xmin
        local_sy = sy + self.y_pos - self.xy_size//2 - ymin
        _seeds[local_sz, local_sx, local_sy] = seeds[sz, sx, sy]

        # tobe refactored watershed segmentation
        _raw = gaussian(_raw / _raw.max(), 2.0)
        _seg = watershed(_raw, markers=_seeds, compactness=0.001)

        # copy unique labels in the source data
        _seg += self.data[segmentation_key].max() + 1
        self.data[segmentation_key][bbox_slices][_mask] = _seg[_mask].ravel()

    def mark_label_ok(self, z, x, y):
        # create bbox from mask
        label_idx = self.cropped_data[segmentation_key][int(z), int(x), int(y)]
        current_status = self.cropped_data[seg_correct_key][int(z), int(x), int(y)]
        mask = self.data[segmentation_key] == label_idx
        self.data[seg_correct_key][mask] = 1 if current_status == 0 else 0

    def clean_seeds(self):
        self.data[seeds_split_key][self.get_slices()] = 0

    def _save_old(self, _slice):
        self.last_slices = _slice
        self.last_seg = np.copy(self.data[segmentation_key][_slice])

    def load_old(self):
        self.data[segmentation_key][self.last_slices] = self.last_seg

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
                print('Label saved')

            @viewer.bind_key('J')
            def _update_boundaries(viewer):
                """Update boundaries"""
                self.update_boundary()
                self.crop_update(viewer)

            @viewer.bind_key('K')
            def _update_segmentation(viewer):
                """Update Segmentation under cursor"""
                z, x, y = viewer.layers[segmentation_key].coordinates
                self.update_segmentation(z, x, y)
                self.update_boundary()
                self.crop_update(viewer)

            @viewer.bind_key('M')
            def _seeds_merge(viewer):
                """Merge label from seeds"""
                points = viewer.layers[seeds_merge_key].data
                self.merge_from_seeds(points)
                self.update_boundary()
                self.crop_update(viewer)
                viewer.layers[seeds_merge_key].data = np.empty((0, 3))

            @viewer.bind_key('N')
            def _seeds_split(viewer):
                """Split label from seeds"""
                seeds = viewer.layers[seeds_split_key].data
                self.split_from_seeds(seeds)
                self.update_boundary()
                self.crop_update(viewer)

            @viewer.bind_key('Control-B')
            def _undo_seeds_split(viewer):
                """Undo-Split label from seeds or Undo-Merge label from seeds"""
                self.load_old()
                self.update_boundary()
                self.crop_update(viewer)

            @viewer.bind_key('C')
            def _clean_split_seeds(viewer):
                """Clean split seeds layer"""
                self.clean_seeds()
                self.crop_update(viewer)

            @viewer.bind_key('O')
            def _seg_correct(viewer):
                z, x, y = viewer.layers[segmentation_key].coordinates
                self.mark_label_ok(z, x, y)
                self.crop_update(viewer)

            @viewer.bind_key('Control-=')
            def zoom_in(viewer):
                """zoom in"""
                self.xy_size = int(self.xy_size * 1.25)
                self.crop_update(viewer)

            @viewer.bind_key('Control--')
            def zoom_out(viewer):
                """zoom out"""
                self.xy_size = int(self.xy_size / 1.25)
                self.crop_update(viewer)


if __name__ == '__main__':
    # 2D example
    BasicProofread(path_raw="/home/lcerrone/datasets/hypocotyl/train/0_19-0521-21_3.h5", z_size=3)()
    # 3D example
    #BasicProofread(path_raw="/home/lcerrone/datasets/small_samples/sample_ovules.h5", z_size=100, xy_size=400)()
