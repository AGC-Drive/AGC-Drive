# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Basedataset class for all kinds of fusion.
"""

import os
import math
from collections import OrderedDict
import random

import torch
import numpy as np
from torch.utils.data import Dataset

import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2

class BaseDataset(Dataset):
    """
    Base dataset for all kinds of fusion. Mainly used to initialize the
    database and associate the __get_item__ index with the correct timestamp
    and scenario.

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    visualize : false
        If set to true, the raw point cloud will be saved in the memory
        for visualization.

    Attributes
    ----------
    scenario_database : OrderedDict
        A structured dictionary contains all file information.

    len_record : list
        The list to record each scenario's data length. This is used to
        retrieve the correct index during training.

    pre_processor : opencood.pre_processor
        Used to preprocess the raw data.

    post_processor : opencood.post_processor
        Used to generate training labels and convert the model outputs to
        bbx formats.

    data_augmentor : opencood.data_augmentor
        Used to augment data.

    """

    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = None
        self.post_processor = None
        self.data_augmentor = DataAugmentor(params['data_augment'], train)

        if 'wild_setting' in params:
            self.seed = params['wild_setting']['seed']
            self.async_flag = params['wild_setting']['async']
            self.async_mode = 'sim' if 'async_mode' not in params['wild_setting'] else params['wild_setting']['async_mode']
            self.async_overhead = params['wild_setting']['async_overhead']
            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']
            self.data_size = params['wild_setting'].get('data_size', 0)
            self.transmission_speed = params['wild_setting'].get('transmission_speed', 27)
            self.backbone_delay = params['wild_setting'].get('backbone_delay', 0)
        else:
            self.async_flag = False
            self.async_overhead = 0
            self.async_mode = 'sim'
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0
            self.transmission_speed = 27
            self.backbone_delay = 0

        root_dir = params['root_dir'] if self.train else params['validate_dir']
        self.max_cav = params.get('train_params', {}).get('max_cav', 7)

        scenario_folders = sorted([os.path.join(root_dir, x) for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))])
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

        for (_, scenario_folder) in enumerate(scenario_folders):
            if not os.path.isdir(os.path.join(scenario_folder, 'cooperative')):
                continue
            scenario_cav_database = OrderedDict()

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

            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs')
                    break
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


    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

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
            # print(idx)
            if idx < ele:
                scenario_index = i
                break
        # print(len(self.scenario_database), scenario_index)
        # print(type(self.scenario_database))
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
        # calculate distance to ego for each cav
        ego_cav_content = \
            self.calc_dist_to_ego(scenario_database, timestamp_key)

        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']
            # data[cav_id]['vehicle_id'] = cav_id

            # calculate delay for this vehicle
            timestamp_delay = \
                self.time_delay_calculation(cav_content['ego'])

            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index
            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = self.return_timestamp_key(scenario_database,
                                                            timestamp_index_delay)
            # add time delay to vehicle parameters
            data[cav_id]['time_delay'] = timestamp_delay
            # load the corresponding data into the dictionary
            data[cav_id]['params'] = self.reform_param(cav_content,
                                                       ego_cav_content,
                                                       timestamp_key,
                                                       timestamp_key_delay,
                                                       cur_ego_pose_flag)
            data[cav_id]['lidar_np'] = \
                pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]['lidar'])
        return data

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
            # in the real mode, time delay = systematic async time + data
            # transmission time + backbone computation time
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

    def reform_param(self, cav_content, ego_content, timestamp_cur,
                     timestamp_delay, cur_ego_pose_flag):
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
        # print(cur_params)

        cur_ego_params = load_yaml(ego_content[timestamp_cur]['yaml'])
        delay_ego_params = load_yaml(ego_content[timestamp_delay]['yaml'])

        # we need to calculate the transformation matrix from cav to ego
        # at the delayed timestamp
        delay_cav_lidar_pose = delay_params['lidar_pose']
        delay_ego_lidar_pose = delay_ego_params["lidar_pose"]

        cur_ego_lidar_pose = cur_ego_params['lidar_pose']
        cur_cav_lidar_pose = cur_params['lidar_pose']

        if not cav_content['ego'] and self.loc_err_flag:
            delay_cav_lidar_pose = self.add_loc_noise(delay_cav_lidar_pose,
                                                      self.xyz_noise_std,
                                                      self.ryp_noise_std)
            cur_cav_lidar_pose = self.add_loc_noise(cur_cav_lidar_pose,
                                                    self.xyz_noise_std,
                                                    self.ryp_noise_std)

        if cur_ego_pose_flag:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             delay_ego_lidar_pose)
            spatial_correction_matrix = x1_to_x2(delay_ego_lidar_pose,
                                                 cur_ego_lidar_pose)
        # This is only used for late fusion, as it did the transformation
        # in the postprocess, so we want the gt object transformation use
        # the correct one
        gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose,
                                            cur_ego_lidar_pose)

        
        # we always use current timestamp's gt bbx to gain a fair evaluation
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
                delay_params['transformation_matrix'] = np.array(delay_json[1]["adjusted_pairwise_t_matrix2"])
                delay_params['gt_transformation_matrix'] = np.array(cur_json[1]["adjusted_pairwise_t_matrix2"])
            else:
                delay_params['transformation_matrix'] = np.array(delay_json[0]["adjusted_pairwise_t_matrix1"])
                delay_params['gt_transformation_matrix'] = np.array(cur_json[0]["adjusted_pairwise_t_matrix1"])
        delay_params['spatial_correction_matrix'] = spatial_correction_matrix

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
                                    timestamp + '_camera0.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + '_camera1.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + '_camera2.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + '_camera3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]

    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points
            with shape (img_row, img_col).

        """
        return self.pre_processor.project_points_to_bev_map(points, ratio)

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask

    def collate_batch_train(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for early and late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # during training, we only care about ego.
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        label_dict_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            label_dict_list.append(ego_dict['label_dict'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(processed_lidar_list)
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'label_dict': label_torch_dict})
        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def visualize_result(self, pred_box_tensor,
                         gt_tensor,
                         pcd,
                         show_vis,
                         save_path,
                         dataset=None):
        # visualize the model output
        self.post_processor.visualize(pred_box_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis,
                                      save_path,
                                      dataset=dataset)
