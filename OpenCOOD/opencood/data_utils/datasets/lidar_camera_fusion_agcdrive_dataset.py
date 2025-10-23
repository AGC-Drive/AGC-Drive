# -*- coding: utf-8 -*-
# Author: Binyu Zhao <byzhao@stu.hit.edu>
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for intermediate fusion with lidar-camera
"""
import os
import math
import bisect
import torch
import random
import numpy as np
from PIL import Image
from collections import OrderedDict

import opencood.data_utils.datasets
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
import opencood.data_utils.post_processor as post_processor
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import box_utils, pcd_utils, transformation_utils, camera_utils, sensor_transformation_utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev


def merge_features_to_dict(processed_feature_list, merge=None):
    """
    Merge the preprocessed features from different cavs to the same
    dictionary.

    Parameters
    ----------
    processed_feature_list : list
        A list of dictionary containing all processed features from
        different cavs.

    Returns
    -------
    merged_feature_dict: dict
        key: feature names, value: list of features.
    """

    merged_feature_dict = OrderedDict()

    for i in range(len(processed_feature_list)):
        for feature_name, feature in processed_feature_list[i].items():

            if feature_name not in merged_feature_dict:
                merged_feature_dict[feature_name] = []
            if isinstance(feature, list):
                merged_feature_dict[feature_name] += feature
            else:
                merged_feature_dict[feature_name].append(feature)

    # stack them
    # it usually happens when merging cavs images -> v.shape = [N, Ncam, C, H, W]
    # cat them
    # it usually happens when merging batches cav images -> v is a list [(N1+N2+...Nn, Ncam, C, H, W))]
    if merge=='stack': 
        for feature_name, features in merged_feature_dict.items():
            merged_feature_dict[feature_name] = torch.stack(features, dim=0)
    elif merge=='cat':
        for feature_name, features in merged_feature_dict.items():
            merged_feature_dict[feature_name] = torch.cat(features, dim=0)

    return merged_feature_dict


def extract_timestamps(yaml_files):
    """
    Given the list of the yaml files, extract the mocked timestamps.

    Parameters
    ----------
    yaml_files : list
        The full path of all yaml files of ego vehicle

    Returns
    -------
    timestamps : list
        The list containing timestamps only.
    """
    timestamps = []

    for file in yaml_files:
        res = file.split('/')[-1]

        timestamp = res.replace('.yaml', '')
        timestamps.append(timestamp)

    return timestamps


def load_camera_files(cav_path, timestamp):
    """
    Retrieve the paths to all camera files.

    Parameters
    ----------
    cav_path : str
        The full file path of current cav.

    timestamp : str
        Current timestamp

    Returns
    -------
    camera_files : list
        The list containing all camera png file paths.
    """
    camera0_file = os.path.join(cav_path, timestamp + '_camera1.png')
    camera1_file = os.path.join(cav_path, timestamp + '_camera3.png')
    camera2_file = os.path.join(cav_path, timestamp + '_camera4.png')
    camera3_file = os.path.join(cav_path, timestamp + '_camera5.png')
    return [camera0_file, camera1_file, camera2_file, camera3_file]


def return_timestamp_key(scenario_database, timestamp_index):
    """
    Given the timestamp index, return the correct timestamp key, e.g.
    2 --> '000078'.
    Parameters
    ----------
    scenario_database : OrderedDict
        The dictionary contains all contents in the current scenario.
    timestamp_index : int
        The index for timestamp.
    Returns
    -------
    timestamp_key : str
        The timestamp key saved in the cav dictionary.
    """
    # get all timestamp keys
    timestamp_keys = list(scenario_database.items())[0][1]
    timestamp_keys = OrderedDict((k, v) for k, v in timestamp_keys.items() if k != 'vehicle_id')
    # retrieve the correct index
    timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

    return timestamp_key


class LiDARCameraIntermediateFusionDataset(torch.utils.data.Dataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.data_augmentor = DataAugmentor(params['data_augment'], train)
        self.pre_processor = build_preprocessor(params['preprocess'], train)
        self.post_processor = post_processor.build_postprocessor(params['postprocess'], dataset='opv2v', train=train)

        if 'train_params' not in params or 'max_cav' not in params['train_params']:
            self.max_cav = 7
        else:
            self.max_cav = params['train_params']['max_cav']
        
        # if project first, cav's lidar will first be projected to the ego's coordinate frame. otherwise, the feature will be projected instead.
        self.proj_first = params['fusion']['args']['proj_first'] if 'proj_first' in params['fusion']['args'] else False
        print('proj_first: ', self.proj_first)
        # whether there is a time delay between the time that cav project lidar to ego and the ego receive the delivered feature
        self.cur_ego_pose_flag = True if 'cur_ego_pose_flag' not in params['fusion']['args'] else params['fusion']['args']['cur_ego_pose_flag']
        
        self.grid_conf = params["fusion"]["args"]["grid_conf"]
        self.depth_discre = camera_utils.depth_discretization(*self.grid_conf['ddiscr'], self.grid_conf['mode'])
        self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # if the training/testing include noisy setting
        if 'wild_setting' in params:
            self.seed = params['wild_setting']['seed']
            # whether to add time delay
            self.async_flag = params['wild_setting']['async']
            self.async_mode = 'sim' if 'async_mode' not in params['wild_setting'] else params['wild_setting']['async_mode']
            self.async_overhead = params['wild_setting']['async_overhead']

            # localization error
            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']

            # transmission data size
            self.data_size = params['wild_setting']['data_size'] if 'data_size' in params['wild_setting'] else 0
            self.transmission_speed = params['wild_setting']['transmission_speed'] if 'transmission_speed' in params['wild_setting'] else 27
            self.backbone_delay = params['wild_setting']['backbone_delay'] if 'backbone_delay' in params['wild_setting'] else 0

        else:
            self.async_flag = False
            self.async_overhead = 0  # ms
            self.async_mode = 'sim'
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0  # Mb (Megabits)
            self.transmission_speed = 27  # Mbps
            self.backbone_delay = 0  # ms

        # build database
        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x) for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))])
        
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path, lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []
        valid_index = 0

        self.uav_flag = False
        self.only_uav_flag = False
        self.agc_drive_dict = {}

        if 'fusion' in params:
            fusion_args = params['fusion']['args']
            self.uav_flag = fusion_args.get('uav_flag', False)
            self.only_uav_flag = fusion_args.get('only_uav_flag', False)
            self.rebuttal_flag = fusion_args.get('rebuttal_flag', False)
            agc_drive_json_path = os.path.join(root_dir, 'AGC-Drive.json')
            if os.path.exists(agc_drive_json_path):
                import json
                with open(agc_drive_json_path, 'r') as f:
                    self.agc_drive_dict = json.load(f)

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(scenario_folders):
            if not os.path.isdir(os.path.join(scenario_folder, 'cooperative')):
                continue
            scenario_cav_database = OrderedDict()
            # self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            # cav_list = sorted([x for x in os.listdir(scenario_folder) if os.path.isdir(os.path.join(scenario_folder, x))])
            # assert len(cav_list) > 0
            # roadside unit data's id is always negative, so here we want to make sure they will be in the end of the list as they shouldn't be ego vehicle.
            # if int(cav_list[0]) < 0:
            #     cav_list = cav_list[1:] + [cav_list[0]]
            if self.uav_flag:
                cav_list = sorted([x for x in os.listdir(scenario_folder) if os.path.isdir(os.path.join(scenario_folder, x)) and x != 'cooperative'])
            else:
                cav_list = sorted([x for x in os.listdir(scenario_folder) if os.path.isdir(os.path.join(scenario_folder, x)) and x != '3' and x != 'cooperative'])

            if len(cav_list) == 0:
                continue
            if self.uav_flag and self.only_uav_flag and len(cav_list) != 3:
                continue
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs')
                    break
                # self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                json_path = os.path.join(scenario_folder, 'cooperative')
                cav_path = os.path.join(scenario_folder, cav_id)
                yaml_files = sorted([os.path.join(json_path, x) for x in os.listdir(json_path) if x.endswith('.json') and 'additional' not in x])
                timestamps = self.extract_timestamps(yaml_files)

                scenario_name = os.path.basename(scenario_folder)
                valid_uav_frames = self.agc_drive_dict.get(scenario_name, [])
                if self.rebuttal_flag:
                    timestamps = [t for t in timestamps if t in valid_uav_frames]
                if cav_id == '3':
                    timestamps = [t for t in timestamps if t in valid_uav_frames]

                if self.uav_flag and self.only_uav_flag:
                    timestamps = [t for t in timestamps if t in valid_uav_frames]
                    if len(timestamps) == 0:
                        continue
                    else:
                        scenario_cav_database[cav_id] = OrderedDict()
                        scenario_cav_database[cav_id]['vehicle_id'] = 'agent' + str(cav_id)
                else:
                    scenario_cav_database[cav_id] = OrderedDict()
                    scenario_cav_database[cav_id]['vehicle_id'] = 'agent' + str(cav_id)

                for timestamp in timestamps:
                    scenario_cav_database[cav_id][timestamp] = OrderedDict()
                    json_file = os.path.join(json_path, timestamp + '.json')
                    yaml_file = os.path.join(cav_path, timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path, timestamp + '.pcd')
                    camera_files = self.load_camera_files(cav_path, timestamp)
                    scenario_cav_database[cav_id][timestamp]['json'] = json_file
                    scenario_cav_database[cav_id][timestamp]['yaml'] = yaml_file
                    scenario_cav_database[cav_id][timestamp]['lidar'] = lidar_file
                    scenario_cav_database[cav_id][timestamp]['camera0'] = camera_files
                # Assume all cavs will have the same timestamps length. Thus we only need to calculate for the first vehicle in the scene.
                if j == 0 and len(timestamps) > 0:
                    scenario_cav_database[cav_id]['ego'] = True
                    # if not self.len_record:
                    #     self.len_record.append(len(timestamps))
                    # else:
                    #     prev_last = self.len_record[-1]
                    #     self.len_record.append(prev_last + len(timestamps))
                else:
                    scenario_cav_database[cav_id]['ego'] = False

            # 统一清洗和对齐 timestamps 部分：
            if self.uav_flag and self.only_uav_flag:
                if set(['1', '2', '3']).issubset(set(scenario_cav_database.keys())):
                    # timestamps_list = [set(scenario_cav_database[cav_id].keys()) - {'ego', 'vehicle_id'} for cav_id in ['1', '2', '3']]
                    timestamps_list = [set(scenario_cav_database[cav_id].keys()) - {'ego', 'vehicle_id'} for cav_id in ['1', '3']]
                    common_timestamps = set.intersection(*timestamps_list)
                    if len(common_timestamps) == 0:
                        continue
                    # for cav_id in ['1', '3']:
                    for cav_id in ['1', '3']:
                        timestamps = list(scenario_cav_database[cav_id].keys())
                        for t in timestamps:
                            if t not in common_timestamps and t not in ['ego', 'vehicle_id']:
                                del scenario_cav_database[cav_id][t]
                else:
                    continue
            elif self.uav_flag:
                # 必须至少有车1和车2
                if '1' in scenario_cav_database and '2' in scenario_cav_database:
                    # 提取车车的交集作为基础帧
                    car_timestamps_list = [
                        set(scenario_cav_database[cav_id].keys()) - {'ego', 'vehicle_id'} 
                        for cav_id in ['1', '2']
                    ]
                    base_common_ts = set.intersection(*car_timestamps_list)

                    # 如果有 UAV 数据，就添加上（可选）
                    if '3' in scenario_cav_database:
                        uav_ts = set(scenario_cav_database['3'].keys()) - {'ego', 'vehicle_id'}

                        # 对每帧基础帧：判断是否也有 UAV
                        for t in list(base_common_ts):
                            for cav_id in ['1', '2']:
                                if t not in scenario_cav_database[cav_id]:
                                    continue
                            if t not in uav_ts:
                                # 没有 UAV，这一帧只保留车1、2
                                scenario_cav_database.pop('3', None)  # 如果 UAV 没有这一帧，就删掉 UAV 的 t
                            # else: UAV 有这个 t，可以留下

                        # 只保留车和 UAV 的共同帧（车车为基础，UAV 有则加）
                        expected_cavs = ['1', '2']
                        if '3' in scenario_cav_database:
                            expected_cavs.append('3')

                        # 过滤掉每个 agent 中不在 base_common_ts 的帧
                        for cav_id in expected_cavs:
                            timestamps = list(scenario_cav_database[cav_id].keys())
                            for t in timestamps:
                                if t not in base_common_ts and t not in ['ego', 'vehicle_id']:
                                    del scenario_cav_database[cav_id][t]

                        # 保留 expected_cavs
                        scenario_cav_database = {k: scenario_cav_database[k] for k in expected_cavs}

                    else:
                        # 没有 UAV，车车照常对齐
                        for cav_id in ['1', '2']:
                            timestamps = list(scenario_cav_database[cav_id].keys())
                            for t in timestamps:
                                if t not in base_common_ts and t not in ['ego', 'vehicle_id']:
                                    del scenario_cav_database[cav_id][t]
                        scenario_cav_database = {k: scenario_cav_database[k] for k in ['1', '2']}

                else:
                    # 没有车1+2就不处理
                    continue 
            else:
                if '1' in scenario_cav_database and '2' in scenario_cav_database:
                    expected_cavs = ['1', '2']
                    timestamps_list = [set(scenario_cav_database[cav_id].keys()) - {'ego', 'vehicle_id'} for cav_id in expected_cavs]
                    common_timestamps = set.intersection(*timestamps_list)
                    if len(common_timestamps) == 0:
                        continue
                    for cav_id in expected_cavs:
                        timestamps = list(scenario_cav_database[cav_id].keys())
                        for t in timestamps:
                            if t not in common_timestamps and t not in ['ego', 'vehicle_id']:
                                del scenario_cav_database[cav_id][t]
                    scenario_cav_database = {k: scenario_cav_database[k] for k in expected_cavs}
                else:
                    continue

            self.scenario_database.update({valid_index: scenario_cav_database})

            if len(scenario_cav_database) > 0:
                num_timestamps = len(next(iter(scenario_cav_database.values())).keys()) - 2  # 减去 'ego' 和 'vehicle_id'
                if not self.len_record:
                    self.len_record.append(num_timestamps)
                else:
                    prev_last = self.len_record[-1]
                    self.len_record.append(prev_last + num_timestamps)

            valid_index += 1
        print('dataset length: ', self.len_record[-1])

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx, cur_ego_pose_flag=self.cur_ego_pose_flag)

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert cav_id == list(base_data_dict.keys())[0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        pairwise_t_matrix, img_pairwise_t_matrix = self.get_pairwise_transformation(base_data_dict, self.max_cav)
        
        agents_image_inputs = []
        processed_features = []
        object_stack = []
        object_id_stack = []
        # prior knowledge for time delay correction and indicating data type (V2V vs V2i)
        velocity = []
        time_delay = []
        infra = []
        spatial_correction_matrix = []

        if self.visualize:
            projected_lidar_stack = []

        lidar_pose = []
        ego_flag = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 + 
                                 (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose, cav_id)

            object_id_stack += selected_cav_processed['object_ids']
            object_stack.append(selected_cav_processed['object_bbx_center'])
            processed_features.append(selected_cav_processed['processed_features'])
            agents_image_inputs.append(selected_cav_processed['image_inputs'])

            velocity.append(selected_cav_processed['velocity'])
            time_delay.append(float(selected_cav_base['time_delay']))
            # this is only useful when proj_first = True, and communication delay is considered. Right now only V2X-ViT utilizes the
            # spatial_correction. There is a time delay when the cavs project their lidar to ego and when the ego receives the feature, and
            # this variable is used to correct such pose difference (ego_t-1 to ego_t)
            spatial_correction_matrix.append(selected_cav_base['params']['spatial_correction_matrix'])
            infra.append(1 if int(cav_id) < 0 else 0)

            lidar_pose.append(selected_cav_base['params']['lidar_pose'])
            if ego_id == cav_id:
                ego_flag.append(True)
            else:
                ego_flag.append(False)

            if self.visualize:
                projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
        # merge preprocessed features from different cavs into the same dict
        merged_feature_dict = merge_features_to_dict(processed_features)
        merged_image_inputs_dict = merge_features_to_dict(agents_image_inputs, merge='stack')


        # exclude all repetitive objects
        unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1



        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center,anchors=anchor_box,mask=mask)

        # pad dv, dt, infra to max_cav
        velocity = velocity + (self.max_cav - len(velocity)) * [0.]
        time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
        infra = infra + (self.max_cav - len(infra)) * [0.]
        spatial_correction_matrix = np.stack(spatial_correction_matrix)
        padding_eye = np.tile(np.eye(4)[None],(self.max_cav - len(spatial_correction_matrix),1,1))
        spatial_correction_matrix = np.concatenate([spatial_correction_matrix, padding_eye], axis=0)
        
        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {
            'cav_num': len(processed_features),
            'lidar_pose': lidar_pose,
            'img_pairwise_t_matrix': img_pairwise_t_matrix,
            'pairwise_t_matrix': pairwise_t_matrix,
            'spatial_correction_matrix': spatial_correction_matrix,
            'image_inputs': merged_image_inputs_dict,
            'processed_lidar': merged_feature_dict,
            'label_dict': label_dict,
            'anchor_box': anchor_box,
            'object_ids': [object_id_stack[i] for i in unique_indices],
            'object_bbx_center': object_bbx_center,
            'object_bbx_mask': mask,

            'velocity': velocity,
            'time_delay': time_delay,
            'infra': infra,
            'ego_flag': ego_flag
        }

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar': np.vstack(projected_lidar_stack)})
        return processed_data_dict

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix. If set to false, meaning when other cavs
            project their LiDAR point cloud to ego, they are projecting to
            past ego pose.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = return_timestamp_key(scenario_database, timestamp_index)
        # calculate distance to ego for each cav
        ego_cav_content = self.calc_dist_to_ego(scenario_database, timestamp_key)

        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']

            # calculate delay for this vehicle
            timestamp_delay = self.time_delay_calculation(cav_content['ego'])

            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index
            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = return_timestamp_key(scenario_database, timestamp_index_delay)

            # add time delay to vehicle parameters
            data[cav_id]['time_delay'] = timestamp_delay
            # load the corresponding data into the dictionary
            data[cav_id]['params'] = self.reform_param(cav_content, ego_cav_content, timestamp_key, timestamp_key_delay, cur_ego_pose_flag)
            data[cav_id]['lidar_np'] = pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]['lidar'])
            img_src = []
            for idx in range(self.data_aug_conf['Ncams']):
                image_path = cav_content[timestamp_key_delay]['camera'][idx]
                img_src.append(Image.open(image_path))
            data[cav_id]['camera_data'] = img_src
        return data

    def calc_dist_to_ego(self, scenario_database, timestamp_key):
        """
        Calculate the distance to ego for each cav.
        """
        ego_lidar_pose = None
        ego_cav_content = None
        for cav_id, cav_content in scenario_database.items():
            if cav_content['ego']:
                ego_cav_content = cav_content
                # print(f"Debug: timestamp_key = {timestamp_key}")
                assert cav_content[timestamp_key] is not None
                assert not isinstance(cav_content[timestamp_key], str), f"{timestamp_key}的数据类型是 {type(cav_content[timestamp_key])}, 内容是：{cav_content[timestamp_key]}"
                # print(f"Debug: cav_content[{timestamp_key}] = {type(cav_content[timestamp_key])}")
                # print(f"Debug: cav_content[{timestamp_key}] = {cav_content[timestamp_key]}")
                # print(f"Debug: 尝试加载 YAML from {cav_content[timestamp_key].get('yaml', '无 yaml 键')}")
                # print(load_yaml(cav_content[timestamp_key]['yaml']))
                ego_lidar_pose = load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
                break
            
        assert ego_lidar_pose is not None
    
        for cav_id, cav_content in scenario_database.items():
            if timestamp_key not in cav_content or not isinstance(cav_content[timestamp_key], dict):
                continue  # 跳过像 'vehicle_id' 或无效时间戳的键
            cur_lidar_pose = load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
            distance = math.sqrt((cur_lidar_pose[0] - ego_lidar_pose[0]) ** 2 +
                                 (cur_lidar_pose[1] - ego_lidar_pose[1]) ** 2)
            cav_content['distance_to_ego'] = distance
            scenario_database.update({cav_id: cav_content})
    
        return ego_cav_content

    def time_delay_calculation(self, ego_flag):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : int
            The time delay quantization.
        """
        # there is not time delay for ego vehicle
        if ego_flag:
            return 0
        # time delay real mode
        if self.async_mode == 'real':
            # in the real mode, time delay = systematic async time + data transmission time + backbone computation time
            overhead_noise = np.random.uniform(0, self.async_overhead)
            tc = self.data_size / self.transmission_speed * 1000
            time_delay = int(overhead_noise + tc + self.backbone_delay)
        elif self.async_mode == 'sim':
            # in the simulation mode, the time delay is constant
            time_delay = np.abs(self.async_overhead)

        # the data is 10 hz for both opv2v and v2x-set
        # todo: it may not be true for other dataset like DAIR-V2X and V2X-Sim
        time_delay = time_delay // 100
        return time_delay if self.async_flag else 0

    def add_loc_noise(self, pose, xyz_std, ryp_std):
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : list
            x,y,z,roll,yaw,pitch

        xyz_std : float
            std of the gaussian noise on xyz

        ryp_std : float
            std of the gaussian noise
        """
        np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, xyz_std, 3)
        ryp_std = np.random.normal(0, ryp_std, 3)
        noise_pose = [pose[0] + xyz_noise[0],
                      pose[1] + xyz_noise[1],
                      pose[2] + xyz_noise[2],
                      pose[3],
                      pose[4] + ryp_std[1],
                      pose[5]]
        return noise_pose

    def reform_param(self, cav_content, ego_content, timestamp_cur, timestamp_delay, cur_ego_pose_flag):
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose for other CAVs.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.

        ego_content : dict
            Ego vehicle content.

        timestamp_cur : str
            The current timestamp.

        timestamp_delay : str
            The delayed timestamp.

        cur_ego_pose_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        Return
        ------
        The merged parameters.
        """        
        cur_params = load_yaml(cav_content[timestamp_cur]['yaml'])
        delay_params = load_yaml(cav_content[timestamp_delay]['yaml'])

        cur_json = load_yaml(cav_content[timestamp_cur]['json'])
        delay_json = load_yaml(cav_content[timestamp_delay]['json'])

        cur_ego_params = load_yaml(ego_content[timestamp_cur]['yaml'])
        delay_ego_params = load_yaml(ego_content[timestamp_delay]['yaml'])

        # we need to calculate the transformation matrix from cav to ego
        # at the delayed timestamp
        delay_cav_lidar_pose = delay_params['lidar_pose']
        delay_ego_lidar_pose = delay_ego_params["lidar_pose"]

        cur_ego_lidar_pose = cur_ego_params['lidar_pose']
        cur_cav_lidar_pose = cur_params['lidar_pose']

        if not cav_content['ego'] and self.loc_err_flag:
            delay_cav_lidar_pose = self.add_loc_noise(delay_cav_lidar_pose, self.xyz_noise_std, self.ryp_noise_std)
            cur_cav_lidar_pose = self.add_loc_noise(cur_cav_lidar_pose, self.xyz_noise_std, self.ryp_noise_std)
        
        if cur_ego_pose_flag:
            transformation_matrix = transformation_utils.x1_to_x2(delay_cav_lidar_pose, cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = transformation_utils.x1_to_x2(delay_cav_lidar_pose, delay_ego_lidar_pose)
            spatial_correction_matrix = transformation_utils.x1_to_x2(delay_ego_lidar_pose, cur_ego_lidar_pose)
        
        # This is only used for late fusion, as it did the transformation in the postprocess, so we want the gt object transformation use the correct one
        gt_transformation_matrix = transformation_utils.x1_to_x2(cur_cav_lidar_pose, cur_ego_lidar_pose)

        vehicles = {}
        for item in cur_json:
            if not isinstance(item, dict):
                continue
            if 'objects' in item and item['objects'] is not None:
                parsed_vehicles = self.parse_objects_to_vehicles(item['objects'])
                vehicles.update(parsed_vehicles)
        # delay_params['vehicles'] = vehicles
        # delay_params['vehicles'] = self.parse_objects_to_vehicles(cur_json[-1]['objects'])
        if cav_content['ego']:
            delay_params['vehicles'] = vehicles
            delay_params['transformation_matrix'] = transformation_matrix
            delay_params['gt_transformation_matrix'] = \
                gt_transformation_matrix
        else:
            delay_params['vehicles'] = vehicles
            if self.uav_flag and str(cav_content['vehicle_id']) == 'agent3':
                delay_params['transformation_matrix'] = np.array(delay_json[1]["pairwise_t_matrix2"])
                delay_params['gt_transformation_matrix'] = np.array(cur_json[1]["pairwise_t_matrix2"])
            else:
                delay_params['transformation_matrix'] = np.array(delay_json[0]["pairwise_t_matrix1"])
                delay_params['gt_transformation_matrix'] = np.array(cur_json[0]["pairwise_t_matrix1"])
        delay_params['spatial_correction_matrix'] = spatial_correction_matrix
        
        camera_to_lidar_matrix, camera_intrinsic = [], []
        for idx in range(self.data_aug_conf['Ncams']):
            camera_id = self.data_aug_conf['cams'][idx]

            camera_pos = delay_params[camera_id]['cords']
            lidar_pose = delay_params['lidar_pose']
            camera2lidar = transformation_utils.x1_to_x2(camera_pos, lidar_pose)
            extrinsic = delay_params[camera_id]['extrinsic']
            # print(camera2lidar, extrinsic)
            # camera_to_lidar_matrix.append(delay_params[camera_id]['extrinsic'])
            camera_to_lidar_matrix.append(extrinsic)
            camera_intrinsic.append(delay_params[camera_id]['intrinsic'])

        delay_params['camera2lidar_matrix'] = camera_to_lidar_matrix
        delay_params['camera_intrinsic'] = camera_intrinsic
        
        return delay_params
    
    @staticmethod
    def parse_objects_to_vehicles(objects):
        # print(objects)
        vehicles = {}
        
        for obj in objects:
            if not (
                (obj.get('className') and obj.get('className').lower() == 'car') or
                (not obj.get('className') and obj.get('modelClass') and obj.get('modelClass').lower() == 'car')
            ):
                continue
            if obj.get('type') != '3D_BOX':
                continue
            track_name = obj['trackName']
            if track_name is None:
                track_name = str(random.randint(100, 999))
            contour = obj['contour']
            if contour is None:
                continue  # 或者打个 warning log，或者根据需要补个默认值
            
            rotation = contour['rotation3D']
            center = contour['center3D']
            size = contour['size3D']

            vehicles[int(track_name)] = {
                'angle': [np.degrees(rotation['x']), np.degrees(rotation['z']), np.degrees(rotation['y'])],
                'center': [0, 0, 0],  # 如果 center 相对坐标有需求可以改这里
                'extent': [size['x']/2, size['y']/2, size['z']/2],
                'location': [center['x'], center['y'], center['z']],
                'speed': 0  # 没有速度，默认 0，有的话可以从其他字段取
            }

        return vehicles

    def get_item_single_car(self, selected_cav_base, ego_pose, cav_id=None):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        # Transformation matrix, intrinsic(cam_K)
        camera_to_lidar_matrix = np.array(selected_cav_base['params']['camera2lidar_matrix']).reshape(4,4,4).astype(np.float32)
        # lidar_to_camera_matrix = np.array(selected_cav_base['params']['lidar2camera_matrix']).astype(np.float32)
        camera_intrinsic = np.array(selected_cav_base['params']["camera_intrinsic"]).reshape(4,3,3).astype(np.float32)

        # data augmentation
        # resize, resize_dims, crop, flip, rotate = camera_utils.sample_augmentation(self.data_aug_conf, self.train)
        # img_src, post_rot2, post_tran2 = camera_utils.img_transform(img_src, torch.eye(2), torch.zeros(2), resize=resize, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate)
        # for convenience, make augmentation matrices 3x3
        post_tran = torch.zeros(4,3)
        #post_tran[:2] = post_tran2
        post_rot = torch.eye(3).unsqueeze(0).repeat(4,1,1)
        #post_rot[:2, :2] = post_rot2

        # image resize and normalize
        reH, reW = self.data_aug_conf['final_dim'][0], self.data_aug_conf['final_dim'][1]
        img_src = []
        for img in selected_cav_base["camera_data"]:
            imgH, imgW = img.height, img.width  # 600 800
            resized_img = img.resize((reW, reH))
            img_src.append(camera_utils.normalize_img(resized_img))

        selected_cav_processed = {
            "image_inputs":{
                "imgs": torch.stack(img_src, dim=0), # [N(cam), 3, H, W]
                "intrins": torch.from_numpy(camera_intrinsic),  # 4*3*3
                "extrins": torch.from_numpy(camera_to_lidar_matrix),    # 4*3*3
                "rots": torch.from_numpy(camera_to_lidar_matrix[:, :3, :3]),  # R_wc, we consider world-coord is the lidar-coord
                "trans": torch.from_numpy(camera_to_lidar_matrix[:, :3, 3]),  # T_wc
                "post_rots": post_rot,  # 4*3
                "post_trans": post_tran,    # 4*3*3
            }
        }
        # for key, value in selected_cav_processed['image_inputs'].items():
        #     print(value.shape)

        # process lidar data

        # calculate the transformation matrix
        transformation_matrix = selected_cav_base['params']['transformation_matrix']

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center([selected_cav_base],ego_pose)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = pcd_utils.shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = pcd_utils.mask_ego_points(lidar_np)

        # project the lidar to ego space
        # xyzi_for_ego = deepcopy(lidar_np)
        if self.proj_first:
            lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
        lidar_np = pcd_utils.mask_points_by_range(lidar_np, self.params['preprocess']['cav_lidar_range'])
        processed_lidar = self.pre_processor.preprocess(lidar_np)

        # ============================================================================================================
        # project pcs to img view, create depth map for self
        depth_maps = []
        for idx, camera_data in enumerate(selected_cav_base["camera_data"]):
            depth_maps.append(self.generate_depth_map(idx, np.array(camera_data), lidar_np, camera_intrinsic[idx], camera_to_lidar_matrix[idx], imgH, imgW, draws=False))
        # ============================================================================================================
        # create depth map for ego
        #xyzi_for_ego[:, :3] = projected_lidar[:,:3]
        #int_matrix_for_ego = np.array(ego_cav_base['params']["camera_intrinsic"]).reshape(4,3,3).astype(np.float32)
        #ext_matrix_for_ego = np.array(ego_cav_base['params']['camera2lidar_matrix']).reshape(4,4,4).astype(np.float32)
        #depth_maps_for_ego = []
        #for idx, camera_data in enumerate(selected_cav_base["camera_data"]):
        #    depth_maps_for_ego.append(self.generate_depth_map(idx, np.array(camera_data), xyzi_for_ego, int_matrix_for_ego[idx], ext_matrix_for_ego[idx], imgH, imgW, draws=False))
        #depth_maps = np.array(depth_maps)
        #depth_map = torch.stack([depth_maps, np.array(depth_maps_for_ego)])  # torch.Size([2, 1, 360, 480])
        # ============================================================================================================
        depth_maps = torch.concat(depth_maps, dim=0)
        selected_cav_processed["image_inputs"].update({"depth_map": depth_maps})

        # velocity
        velocity = selected_cav_base['params']['ego_speed']
        # normalize veloccity by average speed 30 km/h
        velocity = velocity / 30

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': lidar_np,
             'processed_features': processed_lidar,
             'velocity': velocity})

        return selected_cav_processed

    def generate_depth_map(self, index, image, xyz, int_matrix, ext_matrix, imgH, imgW, draws=False):

        blank_image = np.zeros(image.shape, dtype=np.float32)
        rgb_image, points_2d = sensor_transformation_utils.project_lidar_to_camera(index, rgb_image=blank_image, point_cloud=xyz, camera_intrinsic=int_matrix, image_size=(imgH, imgW))
        
        depth = points_2d[:, 2]
        #print('depth: ', np.max(depth), np.min(depth))
        uv = points_2d[:, :2]
        # uv_int = uv.round().astype(np.int32)
        uv_int = (np.ceil(uv) - ((uv - np.floor(uv)) < 0.5).astype(np.int32)).astype(np.int32)
        uv_int = uv_int[:, ::-1]

        valid_mask = ((depth >= self.grid_conf['ddiscr'][0]) &
                      (uv_int[:,0] >= 0) & (uv_int[:,0] < imgH) & 
                      (uv_int[:,1] >= 0) & (uv_int[:,1] < imgW))
        valid_uvint, valid_depth = uv_int[valid_mask], depth[valid_mask]

        depth_map = -1 * np.ones((imgH, imgW), dtype=np.float32)
        for idx, valid_coord in enumerate(valid_uvint):
            u, v = valid_coord[0], valid_coord[1]
            depth_level = bisect.bisect_left(self.depth_discre, valid_depth[idx])
            if depth_level == 0:
                depth_level == 1 
            depth_map[u,v] = depth_level-1 if depth_map[u,v]<0 else min(depth_map[u,v], depth_level-1)
            #depth_map[u,v] = valid_depth[idx] if depth_map[u,v]<0 else min(depth_map[u,v], valid_depth[idx])

        # resize as config requires
        assert imgH % self.data_aug_conf['final_dim'][0] == 0
        assert imgW % self.data_aug_conf['final_dim'][1] == 0
        scaleH, scaleW = imgH // self.data_aug_conf['final_dim'][0], imgW // self.data_aug_conf['final_dim'][1]

        max_depth_level = np.max(depth_map)
        depth_map[depth_map<0] = max_depth_level + 1
        depth_map = torch.FloatTensor(-1 * depth_map).unsqueeze(0)
        pool_layer = torch.nn.MaxPool2d(kernel_size=(scaleH, scaleW), stride=(scaleH, scaleW))
        depth_map = -1 * pool_layer(depth_map)
        depth_map[depth_map>max_depth_level] = -1

        if draws:
            plt.imshow(image)
            plt.savefig(f'image{index}.png')
            plt.close()
            
            plt.imshow(rgb_image)
            plt.savefig(f'proj_image{index}.png')
            plt.close()
            
            plt.imshow(depth_map.numpy().transpose(1,2,0))
            plt.savefig(f'depth{index}.png')
            plt.close()
        
        return depth_map

    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two

        record_len = [] # used to record different scenario
        lidar_poses = []

        img_pairwise_t_matrix_list = [] # pairwise transformation matrix for image
        pairwise_t_matrix_list = [] # pairwise transformation matrix
        processed_lidar_list = []
        image_inputs_list = []
        label_dict_list = []

        object_ids = []
        object_bbx_center = []
        object_bbx_mask = []

        # used for PriorEncoding for models
        velocity = []
        time_delay = []
        infra = []

        # used for correcting the spatial transformation between delayed timestamp
        # and current timestamp
        spatial_correction_matrix_list = []
        ego_flag = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            record_len.append(ego_dict['cav_num'])
            lidar_poses.append(ego_dict['lidar_pose'])
            img_pairwise_t_matrix_list.append(ego_dict['img_pairwise_t_matrix'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            image_inputs_list.append(ego_dict['image_inputs']) # different cav_num, ego_dict['image_inputs'] is dict.
            label_dict_list.append(ego_dict['label_dict'])
            object_ids.append(ego_dict['object_ids'])
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])


            velocity.append(ego_dict['velocity'])
            time_delay.append(ego_dict['time_delay'])
            infra.append(ego_dict['infra'])
            spatial_correction_matrix_list.append(ego_dict['spatial_correction_matrix'])

            ego_flag.append(ego_dict['ego_flag'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])
        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = self.pre_processor.collate_batch(merged_feature_dict)
        merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')

        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        label_torch_dict = self.post_processor.collate_batch(label_dict_list)

        # (B, max_cav)
        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        spatial_correction_matrix_list = torch.from_numpy(np.array(spatial_correction_matrix_list))
        # (B, max_cav, 3)
        prior_encoding = torch.stack([velocity, time_delay, infra], dim=-1).float()
        # (B, max_cav)
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))
        img_pairwise_t_matrix = torch.from_numpy(np.array(img_pairwise_t_matrix_list))

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict = {
            'ego': {
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': object_bbx_mask,
                'object_ids': object_ids[0],
                'label_dict': label_torch_dict,
                
                'processed_lidar': processed_lidar_torch_dict,
                'image_inputs': merged_image_inputs_dict,
                'record_len': record_len,
                'prior_encoding': prior_encoding,
                'spatial_correction_matrix': spatial_correction_matrix_list,
                'pairwise_t_matrix': pairwise_t_matrix,
                'img_pairwise_t_matrix': img_pairwise_t_matrix,
                'lidar_pose': lidar_poses,
                'ego_flag': ego_flag
            }
        }


        if self.visualize:
            origin_lidar = np.array(pcd_utils.downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)

        # check if anchor box in the batch
        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box': torch.from_numpy(np.array(batch[0]['ego']['anchor_box']))})

        # save the transformation matrix (4, 4) to ego vehicle
        transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
        output_dict['ego'].update({'transformation_matrix': transformation_matrix_torch})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        # pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
        preds = self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        # return pred_box_tensor, pred_score, gt_box_tensor
        return preds + (gt_box_tensor,)
    """
    def get_pairwise_transformation(self, base_data_dict, max_cav):
        
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            pairwise_t_matrix[:, :] = np.identity(4)
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                t_list.append(cav_content['params']['transformation_matrix'])

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i == j:
                        t_matrix = np.eye(4)
                        pairwise_t_matrix[i, j] = t_matrix
                        continue
                    # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                    t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                    pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix
    """
    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        identity_pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))

        
        # if lidar projected to ego first, then the pairwise matrix becomes identity
        identity_pairwise_t_matrix[:, :] = np.identity(4)
        
        # save all transformation matrix in a list in order first.
        t_list = []
        for cav_id, cav_content in base_data_dict.items():
            t_list.append(cav_content['params']['transformation_matrix'])

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                # identity matrix to self
                if i == j:
                    t_matrix = np.eye(4)
                    pairwise_t_matrix[i, j] = t_matrix
                    continue
                # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                pairwise_t_matrix[i, j] = t_matrix

        if self.proj_first:
            return identity_pairwise_t_matrix, pairwise_t_matrix
        else:
            return pairwise_t_matrix, pairwise_t_matrix
        
    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.json', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def load_camera_files(cav_path, timestamp):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path,
                                    timestamp + '_camera2.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + '_camera3.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + '_camera4.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + '_camera5.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]