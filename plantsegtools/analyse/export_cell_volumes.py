import argparse
import csv

import h5py
import numpy as np

MOVIE_CONFIGS = {
    'Marvelous': {
        # mamut file containing lineage graph vertices
        'vertices_file': '/home/common/Datasets/Marion/2017-08-02_17.49.34_stPVB003-2-2xDR5v2_F3_nb25_Marvelous/paintera_final/2017-08-02_17.49.34_stPVB003-2-2xDR5v2_F3_nb25_Marvelous_fused_cropped_export-mamut_FeatureAndTagTable-vertices_new.csv',
        # mamut file containing lineage graph edges
        'edges_file': '/home/common/Datasets/Marion/2017-08-02_17.49.34_stPVB003-2-2xDR5v2_F3_nb25_Marvelous/paintera_final/2017-08-02_17.49.34_stPVB003-2-2xDR5v2_F3_nb25_Marvelous_fused_cropped_export-mamut_FeatureAndTagTable-edges_new.csv',
        # volume size of the original movie timepoints
        'volume_size': (486, 1050, 2048),
        # voxel size in microns
        'voxel_size': [0.25, 0.1625, 0.1625],
        # information about the subvolume cropped from the original: those are the volumes used for PlantSeg prediction
        'crop': (slice(0, 486), slice(0, 620), slice(420, 420 + 1330)),
        # timepoints to process
        'tps': ['00', '02', '04', '06', '08', '10', '12', '14', '16', '18', '20'],
        # template path to the predictions
        'file_fun': lambda
            tp: f'/home/common/Datasets/Marion/2017-08-02_17.49.34_stPVB003-2-2xDR5v2_F3_nb25_Marvelous/paintera_final/corrected_crops/t000{tp}_s00_uint8_crop_x420_y620_predictions.h5'
    },
    'Beautiful': {
        'vertices_file': '/home/common/Datasets/Marion/2017-07-31_16.38.32_stPVB003-2-2xDR5v2_F3_nb25_bioutifoul/BDV/2017-09-02_18_43_17_stPVB003-2-2xDR5v2_F3_nb25_bioutifoul_fused_dc_cropped_export-mamut_FeatureAndTagTable-vertices.csv',
        'edges_file': '/home/common/Datasets/Marion/2017-07-31_16.38.32_stPVB003-2-2xDR5v2_F3_nb25_bioutifoul/BDV/2017-09-02_18_43_17_stPVB003-2-2xDR5v2_F3_nb25_bioutifoul_fused_dc_cropped_export-mamut_FeatureAndTagTable-edges.csv',
        'volume_size': (403, 1396, 1940),
        'voxel_size': [0.25, 0.1625, 0.1625],
        'crop': (slice(0, 403), slice(520, 970), slice(40, 1630)),
        'tps': ['00', '02', '04', '06', '08', '10', '12', '14'],
        'file_fun': lambda
            tp: f'/home/common/Datasets/Marion/2017-07-31_16.38.32_stPVB003-2-2xDR5v2_F3_nb25_bioutifoul/paintera_final/corrected_crops/fused_dc_cropped--C00--T000{tp}_crop_x40-1630_y520-970_predictions.h5'
    },
    'Nice': {
        'vertices_file': '/home/common/Datasets/Marion/2017-09-02_18.43.17_stPVB003-2-2xDR5v2_F3_nb25_niice/BDV/2017-09-02_18_43_17_stPVB003-2-2xDR5v2_F3_nb25_fused_cropped_export-mamut_FeatureAndTagTable-vertices.csv',
        'edges_file': '/home/common/Datasets/Marion/2017-09-02_18.43.17_stPVB003-2-2xDR5v2_F3_nb25_niice/BDV/2017-09-02_18_43_17_stPVB003-2-2xDR5v2_F3_nb25_fused_cropped_export-mamut_FeatureAndTagTable-edges.csv',
        # swap the X and Y axes in Mamut coordinates
        'swap_xy_coord': True,
        'volume_size': (566, 2048, 1195),
        'voxel_size': [0.25, 0.1625, 0.1625],
        'crop': (slice(100, 500), slice(410, 1580), slice(430, 790)),
        'tps': ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16',
                '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33',
                '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62'],
        'file_fun': lambda
            tp: f'/home/common/Datasets/Marion/2017-09-02_18.43.17_stPVB003-2-2xDR5v2_F3_nb25_niice/plantseg_final/Cells/niice_C00_T000{tp}_crop_x430-790_y410-1580_z100-500_predictions.h5'

    }
}


class AbstractTrackingResults:
    def get_tracked_spots(self, frame, label_volume):
        """
        Get all the spots from a given time frame and their corresponding labels.

        Args:
            frame(int): frame number
            label_volume(ndarray): segmentation volume

        Returns:
            list of spot objects
        """
        raise NotImplementedError

    def find_parent_id(self, spot, previous_frame, previous_volume):
        """
        Args:
             label(int): label id of the cell for which we want to find a parent
             previous_frame(int): previous frame where the parent should be looked for
             previous_volume(ndarray): segmentation volume from the 'previous_frame'
        Returns:
            ID of a parent spot (string), or None if the parent could not be found
        """
        raise NotImplementedError


class MamutTrackingResults(AbstractTrackingResults):
    """
    Parses Mamut XML files containing description of the lineage graph and combines it with the segmentation results.

    """

    def __init__(self, vertices_file, edges_file, volume_size, voxel_size, crop, swap_xy_coord=False, **kwargs):
        self.crop = crop
        self.volume_size = volume_size
        self.voxel_size = np.array(voxel_size)
        self.swap_xy_coord = swap_xy_coord
        with open(vertices_file, 'r') as f:
            v_reader = csv.DictReader(f)
            self.vertices = list(v_reader)

        with open(edges_file, 'r') as f:
            e_reader = csv.DictReader(f)
            self.edges = list(e_reader)

    def _to_zyx_coordinates(self, mamut_coordinates):
        """
        Takes MaMUT coordinates and the size of the voxel and recovers the pixel coordinates.

        Args:
            mamut_coordinates(tuple): tuple of z,y,x spot coordinates taken from mamut.xml

        Returns:
            (z,y,x) tuple of pixel coordinates
        """
        # convert string to double
        mamut_coordinates = np.array(list(map(float, mamut_coordinates)))
        # recover pixel coordinates
        return list(map(int, mamut_coordinates / self.voxel_size))

    def _restore_volume(self, label_volume):
        """
        Gets the label_volume, which is a crop from the original volume and inserts it into the 0-array of the size of original volume
        """
        volume = np.zeros(self.volume_size, dtype='uint32')
        volume[self.crop] = label_volume
        return volume

    def _get_label(self, v, volume):
        """
        Given a spot object 'v' and the segmented 'volume' (original size), returns the label value from the center of the spot
        Args:
            v(spot obj): spot object, i.e. vertex in the lineage graph
            volume(ndarray): array of the size of original volume, containing the segmented volume

        Returns:
             label id (int) from the 'volume' corresponding the center of the spot obj 'v'
        """
        mamut_coordinates = (v['Z'], v['Y'], v['X'])
        if self.swap_xy_coord:
            mamut_coordinates = (v['Z'], v['X'], v['Y'])

        z, y, x = self._to_zyx_coordinates(mamut_coordinates)
        return volume[z, y, x]

    def get_tracked_spots(self, frame, label_volume):
        volume = self._restore_volume(label_volume)

        results = []
        frame_vertices = list(filter(lambda v: int(v['Frame']) == frame, self.vertices))
        for v in frame_vertices:
            label = self._get_label(v, volume)
            if label != 0:
                v['SegmLabel'] = label
                results.append(v)

        return results

    def find_parent_id(self, spot, previous_frame, previous_volume):
        def _find_parent_spot(s):
            if s is None:
                return s

            spot_id = int(spot['ID'])
            # find parent spot id
            parent_spot = list(filter(lambda e: int(e['Target ID']) == spot_id, self.edges))
            if not parent_spot:
                return None
            # return parent spot
            parent_spot_id = int(parent_spot[0]['Source ID'])
            return list(filter(lambda v: int(v['ID']) == parent_spot_id, self.vertices))[0]

        current_frame = int(spot['Frame'])
        assert previous_frame < current_frame

        for _ in range(current_frame - previous_frame):
            spot = _find_parent_spot(spot)

        if spot is None:
            return None
        else:
            volume = self._restore_volume(previous_volume)
            label = self._get_label(spot, volume)
            if label == 0:
                return None
            return f'{movie_name}_frame_{previous_frame}_label_{label}'


def _create_entry(label, volume, frame, movie_name, parent_id, spot):
    spot_id = None
    if spot is not None:
        spot_id = spot['Label']
    entry = {
        'ID': f'{movie_name}_frame_{frame}_label_{label}',
        'SpotID': spot_id,
        'Frame': frame,
        'MovieName': movie_name,
        'Label': label,
        'Volume': volume,
        'ParentID': parent_id
    }
    return entry


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract cell volumes')
    parser.add_argument('--movie-name', type=str, required=True, help='Name of the movie')
    parser.add_argument('--dataset-name', type=str, help='Name for the label dataset inside the H5',
                        default='segmentation')
    args = parser.parse_args()

    movie_name = args.movie_name
    label_ds = args.dataset_name

    assert movie_name in MOVIE_CONFIGS

    config = MOVIE_CONFIGS[movie_name]
    mtr = MamutTrackingResults(**config)

    entries = []
    volumes = {}
    tps = config['tps']

    for i, tp in enumerate(tps):
        in_file = config['file_fun'](tp)
        print(f'Processing {in_file}...')
        with h5py.File(in_file, 'r') as f:
            label = f[label_ds][...]

        frame = int(tp)
        volumes[frame] = label

        tracked_spots = mtr.get_tracked_spots(frame, label)
        tracked_spots_map = dict(map(lambda s: (s['SegmLabel'], s), tracked_spots))
        print(f'Frame: {frame}, labels corresponding to spots: {tracked_spots_map.keys()}')

        ids, counts = np.unique(label, return_counts=True)
        print(f'Labels in volume: {ids}')

        for id, count in zip(ids, counts):
            parent_id = None
            if id in tracked_spots_map and i != 0:
                print(f'{id} found in tracked spots')
                spot = tracked_spots_map[id]
                previous_frame = int(tps[i - 1])
                print(f'Searching for parent in frame {previous_frame}')
                previous_volume = volumes[previous_frame]
                parent_id = mtr.find_parent_id(spot, previous_frame, previous_volume)
                print(f'Found parent_id: {parent_id}')

            entry = _create_entry(id, count, frame, movie_name, parent_id, spot=tracked_spots_map.get(id, None))
            entries.append(entry)

    keys = entries[0].keys()
    output_csv = movie_name + "_cell_volumes.csv"
    with open(output_csv, "w") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(entries)
