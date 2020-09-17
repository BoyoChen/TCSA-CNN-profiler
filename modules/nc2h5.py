import os
import xarray
import pandas as pd
import numpy as np
import h5py
import argparse
from tqdm import tqdm
from collections import defaultdict


def crop_center(image_matrix, crop_width):
    total_width = image_matrix.shape[1]
    start = total_width // 2 - crop_width // 2
    end = start + crop_width
    return image_matrix[:, start:end, start:end, :]


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', help='path to the folder \'TC_data\'', default='../TC_data')
parser.add_argument('--profile_folder')
parser.add_argument('--output', help='path to the output hdf5 file', default='TCIR.h5')
args = parser.parse_args()

# args.output = 'TCSA.h5'
# args.data_folder = '.'
args.profile_folder = 'everydata'

# =======matrix data=========
regions = ['WPAC', 'EPAC', 'ATLN', 'SH', 'CPAC', 'IO']
info_df = pd.DataFrame(columns=['region', 'ID', 'lon', 'lat', 'time', 'Vmax', 'R34', 'MSLP'])

data_matrix = []
for region in regions:
    print(f'Processing region: {region}')
    region_path = os.path.join(args.data_folder, region)
    cyclones = sorted([entry for entry in os.listdir(region_path) if entry.startswith('20')])
    for cyclone in tqdm(cyclones):
        cyclone_path = os.path.join(region_path, cyclone)
        frames = os.listdir(cyclone_path)
        for frame in frames:
            frame_path = os.path.join(cyclone_path, frame)
            frame_data = xarray.open_dataset(frame_path)
            frame_info = pd.Series(frame_data.attrs)
            frame_info.index = ['ID', 'lon', 'lat', 'time', 'Vmax', 'R34', 'MSLP']
            frame_info['region'] = region
            frame_info['R34'] *= 1.852
            info_df.loc[info_df.shape[0]] = frame_info
            # transfer 201*201 data matrix into numpy ndarray
            data_matrix.append(frame_data.to_array().values.transpose([1, 2, 0]))

info_df.sort_values(['region', 'ID', 'time'], inplace=True)
sorted_idx = np.array(info_df.index)
info_df.reset_index(drop=True, inplace=True)
all_data_matrix = np.stack(data_matrix)
del data_matrix

cropped_data_matrix = crop_center(all_data_matrix, 128)
del all_data_matrix

sorted_data_matrix = cropped_data_matrix[sorted_idx]
del cropped_data_matrix

# =======profile data=========
profile_folder = args.profile_folder
profile_dict = defaultdict(dict)

profiles = sorted([entry for entry in os.listdir(profile_folder) if entry.startswith('20')])
for profile in tqdm(profiles):
    profile_path = os.path.join(profile_folder, profile)
    try:
        profile_data = xarray.open_dataset(profile_path)
    except ValueError:
        print(profile)
    ID = profile_data.attrs['TC_ID']
    time = profile_data.attrs['TC_date (yymmddhh)']
    structure_profile = profile_data['Vnew'].values
    profile_dict[ID][time] = structure_profile

profiles_list = []
valid = []
for index, row in info_df.iterrows():
    profile = profile_dict.get(row['ID'], {}).get(row['time'], -np.ones(151))
    profiles_list.append(profile)
    if profile[0] == -1:
        valid.append(False)
    else:
        valid.append(True)

profile_matrix = np.stack(profiles_list)
info_df['valid_profile'] = valid

# =======store data=========
f = h5py.File(args.output, 'w')
f['images'] = sorted_data_matrix
f['structure_profiles'] = profile_matrix
f.close()
info_df.to_hdf(args.output, 'info')
