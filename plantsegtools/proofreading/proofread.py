import os

import napari
import numpy as np
from skimage.measure import label
from skimage.segmentation import find_boundaries, watershed
from skimage.filters import gaussian
from plantsegtools.utils import create_h5, smart_load, load_h5, H5_FORMATS, relabel_segmentation
from plantsegtools.postprocess.seg_nuclei_consistency import get_bbox
from magicgui import magicgui

raw_key = 'raw'
segmentation_key = 'segmentation'
seg_boundaries_key = 'seg-boundaries'
seeds_split_key = 'seeds-split'
seg_correct_key = 'seg-correct-mask'

seg_boundaries_cmap = {0: None,
                       1: (0.05, 0.05, 0.05, 1.)
                       }
seg_correct_cmap = {0: None,
                    1: (0.76388469, 0.02003777, 0.61156412, 1.)
                    }

out_suffix = 'proofread'
zoom_factor = 1.25
pixel_toll = 2
max_height = 60


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
            if stack.ndim == 2:
                stack = stack[None, ...]
            self.data[key] = stack
            shapes.append(stack.shape)

        assert shapes[0] == shapes[1]
        self.shape = shapes[0]

        self.data[segmentation_key] = self.data[segmentation_key].astype('uint32')

        self.datasets[seg_boundaries_key] = (None, seg_boundaries_key)
        self.data[seg_boundaries_key] = self.get_seg_boundary()

        self.datasets[seeds_split_key] = (None, seeds_split_key)
        self.data[seeds_split_key] = np.zeros(shapes[0]).astype('uint32')

        self.datasets[seg_correct_key] = (None, seg_correct_key)
        self.data[seg_correct_key] = self.load_correct_mask()

        self.xy_size = min(xy_size, min(self.shape[1], self.shape[2]))
        self.z_size = min(z_size, self.shape[0])

        self.cropped_data = {}
        self.z_pos = (self.shape[0]) // 2
        self.x_pos = (self.shape[1]) // 2
        self.y_pos = (self.shape[2]) // 2
        self.stride = self.xy_size // 4 if stride is None else stride

        self.get_crop()

        self.last_slices, self.last_seg = None, None

    def load_correct_mask(self):
        seg_path = self.datasets[segmentation_key][0]
        base, ext = os.path.splitext(seg_path)
        seg_correct_mask = None

        if ext in H5_FORMATS:
            seg_correct_mask, _ = load_h5(seg_path, key=seg_correct_key, safe_mode=True)

        seg_correct_mask = np.zeros(self.shape) if seg_correct_mask is None else seg_correct_mask
        return seg_correct_mask.astype('uint8')

    def get_slices(self):
        z_min, z_max = max(self.z_pos - self.z_size // 2, 0), min(self.z_pos + self.z_size // 2, self.shape[0])
        z_max = z_max if z_max - z_min > 0 else 1
        x_min, x_max = max(self.x_pos - self.xy_size // 2, 0), min(self.x_pos + self.xy_size // 2, self.shape[1])
        y_min, y_max = max(self.y_pos - self.xy_size // 2, 0), min(self.y_pos + self.xy_size // 2, self.shape[2])
        return slice(z_min, z_max), slice(x_min, x_max), slice(y_min, y_max)

    def get_crop(self):
        _slices = self.get_slices()
        print(f'current slice: {_slices}')
        for key, value in self.data.items():
            self.cropped_data[key] = value[_slices]

    def update(self, viewer):
        for _layer_key in viewer.layers:
            viewer.layers[_layer_key.name].data = self.cropped_data[_layer_key.name]

    def crop_update(self, viewer):
        self.get_crop()
        self.update(viewer)

    def init_layers(self):
        self.viewer.add_image(self.cropped_data[raw_key], name=raw_key, multiscale=False)
        self.viewer.add_labels(self.cropped_data[segmentation_key], name=segmentation_key, multiscale=False)
        self.viewer.add_labels(self.cropped_data[seg_correct_key],
                               name=seg_correct_key,
                               color=seg_correct_cmap,
                               opacity=1,
                               multiscale=False)
        self.viewer.add_labels(self.cropped_data[seeds_split_key], name=seeds_split_key)
        self.viewer.add_labels(self.cropped_data[seg_boundaries_key],
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
        return find_boundaries(segmentation).astype('uint8')

    def update_boundary(self):
        local_boundary = find_boundaries(self.cropped_data[segmentation_key])
        self.data[seg_boundaries_key][self.get_slices()] = local_boundary

    def update_segmentation(self, z, x, y):
        # create bbox from mask
        label_idx = self.cropped_data[segmentation_key][int(z), int(x), int(y)]
        mask = self.data[segmentation_key] == label_idx
        bbox_slices, _, _, _ = get_bbox(mask, pixel_toll=2)

        _boundaries = self.data[seg_boundaries_key][bbox_slices]
        _mask = self.data[segmentation_key][bbox_slices] == label_idx
        seeds = label(_boundaries, background=1)
        seeds[~_mask] = 0

        _seg = watershed(np.ones_like(seeds), markers=seeds)
        _seg += self.data[segmentation_key].max() + 1
        self.data[segmentation_key][bbox_slices][_mask] = _seg[_mask].ravel()

    def split_from_seeds(self, seeds):
        # find seeds location ad label value
        sz, sx, sy = np.nonzero(seeds)
        all_idx = self.cropped_data[segmentation_key][sz, sx, sy]
        all_idx = np.unique(all_idx)

        # create bbox from mask
        mask = np.logical_or.reduce([self.data[segmentation_key] == label_idx for label_idx in all_idx])
        bbox_slices, z_min, x_min, y_min = get_bbox(mask)
        self._save_old(bbox_slices)

        # load main bbox data
        _mask = np.logical_or.reduce([self.data[segmentation_key][bbox_slices] == label_idx for label_idx in all_idx])
        _raw = self.data[raw_key][bbox_slices]

        # create bbox seeds
        _seeds = np.zeros_like(_mask).astype(np.int64)
        local_sz = sz + self.z_pos - self.z_size//2 - z_min
        local_sx = sx + self.x_pos - self.xy_size//2 - x_min
        local_sy = sy + self.y_pos - self.xy_size//2 - y_min
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

    def relabel_seg(self, bg_id=0):
        """print relabeling segmentation"""
        seg = self.data[segmentation_key]
        bg_mask = seg == bg_id
        new_seg = relabel_segmentation(seg).astype(np.uint16)
        new_seg[bg_mask] = 0
        self.data[segmentation_key] = new_seg

    def save_h5(self):
        seg_path = self.datasets[segmentation_key][0]
        base, ext = os.path.splitext(seg_path)
        seg_path = f'{base}_{out_suffix}{ext}'
        create_h5(seg_path, self.data[segmentation_key], key='label', mode='w')
        create_h5(seg_path, self.data[segmentation_key].astype('uint16'), key='label_uint16')
        create_h5(seg_path, self.data[raw_key], key=raw_key)
        create_h5(seg_path, self.data[seg_correct_key], key=seg_correct_key)
        print('Label saved')

    def __call__(self):
        self.viewer = napari.Viewer()
        self.init_layers()

        @self.viewer.bind_key('Control-Left')
        def move_left(_viewer):
            """move field of view left"""
            self.move(0, -self.stride)
            self.update(_viewer)

        @self.viewer.bind_key('Control-Right')
        def move_right(_viewer):
            """move field of view right"""
            self.move(0, self.stride)
            self.update(_viewer)

        @self.viewer.bind_key('Control-Up')
        def move_up(_viewer):
            """move field of view up"""
            self.move(-self.stride, 0)
            self.update(_viewer)

        @self.viewer.bind_key('Control-Down')
        def move_down(_viewer):
            """move field of view down"""
            self.move(self.stride, 0)
            self.update(_viewer)

        @self.viewer.bind_key('S')
        def save_current(_viewer):
            """save edits on h5 and create a training ready stack"""
            self.relabel_seg()
            self.save_h5()
            self.crop_update(_viewer)

        @magicgui(call_button="Save (S)")
        def save_current_button():
            save_current(self.viewer)

        dock_save = self.viewer.window.add_dock_widget(save_current_button)
        dock_save.setMaximumHeight(max_height)

        @self.viewer.bind_key('N')
        def seeds_merge_split(_viewer):
            """Split label from seeds"""
            seeds = _viewer.layers[seeds_split_key].data
            self.split_from_seeds(seeds)
            self.update_boundary()
            self.crop_update(_viewer)

        @magicgui(call_button="Split / Merge (N)")
        def seed_merge_split_dock():
            seeds_merge_split(self.viewer)
        merge_split_dock = self.viewer.window.add_dock_widget(seed_merge_split_dock)
        merge_split_dock.setMaximumHeight(max_height)

        @self.viewer.bind_key('Control-N')
        def undo_seed_merge_split(_viewer):
            """Undo-Split label from seeds or Undo-Merge label from seeds"""
            self.load_old()
            self.update_boundary()
            self.crop_update(_viewer)

        @magicgui(call_button="Undo - Split / Merge (Ctrl + N)")
        def undo_seed_merge_split_dock():
            undo_seed_merge_split(self.viewer)

        undo_merge_split_dock = self.viewer.window.add_dock_widget(undo_seed_merge_split_dock)
        undo_merge_split_dock.setMaximumHeight(max_height)

        @self.viewer.bind_key('C')
        def clean_split_seeds(_viewer):
            """Clean split seeds layer"""
            self.clean_seeds()
            self.crop_update(_viewer)

        @magicgui(call_button="Clean Seeds (C)")
        def clean_split_seeds_dock():
            clean_split_seeds(self.viewer)

        split_dock = self.viewer.window.add_dock_widget(clean_split_seeds_dock)
        split_dock.setMaximumHeight(max_height)

        @self.viewer.bind_key('O')
        def _seg_correct(_viewer):
            _pos = _viewer.cursor.position
            z, x, y = _viewer.layers[segmentation_key].world_to_data(_pos)
            self.mark_label_ok(z, x, y)
            self.crop_update(_viewer)

        @self.viewer.bind_key('J')
        def update_boundaries(_viewer):
            """Update boundaries"""
            self.update_boundary()
            self.crop_update(_viewer)

        @self.viewer.bind_key('B')
        def show_un_show_corrected_labels(_viewer):
            """Update Segmentation under cursor"""
            _pos = _viewer.cursor.position
            _viewer.layers[seg_correct_key].visible = not _viewer.layers[seg_correct_key].visible

        @magicgui(call_button="Toggle Correct Layer (B)")
        def show_un_show_corrected_labels_dock():
            show_un_show_corrected_labels(self.viewer)

        show_un_show_corrected_labels_dock_wb = self.viewer.window.add_dock_widget(show_un_show_corrected_labels_dock)
        show_un_show_corrected_labels_dock_wb.setMaximumHeight(max_height)

        @self.viewer.bind_key('K')
        def update_segmentation(_viewer):
            """Update Segmentation under cursor"""
            _pos = _viewer.cursor.position
            z, x, y = _viewer.layers[segmentation_key].world_to_data(_pos)
            self.update_segmentation(z, x, y)
            self.update_boundary()
            self.crop_update(_viewer)

        @self.viewer.bind_key('Alt-Up')
        def zoom_in(_viewer):
            """zoom in"""
            self.xy_size = int(self.xy_size * zoom_factor)
            self.crop_update(_viewer)

        @self.viewer.bind_key('Alt-Down')
        def zoom_out(_viewer):
            """zoom out"""
            self.xy_size = int(self.xy_size / zoom_factor)
            self.crop_update(_viewer)

        napari.run()
