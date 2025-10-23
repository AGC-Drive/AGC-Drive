import os
import glob
import numpy as np
import pandas as pd
import yaml
from multiprocessing import Pool

def yaml_safe_dump(data, path):
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float_, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int_, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        else:
            return obj

    clean_data = convert_numpy(data)
    with open(path, 'w') as f:
        yaml.dump(clean_data, f)

def process_scene(args):
    scene_dir, new_scene_dir = args
    print(f"Processing {scene_dir}")

    fragment_dirs = glob.glob(os.path.join(scene_dir, '*'))
    for fragment in fragment_dirs:
        gps_dir = os.path.join(fragment, 'gps')
        imu_dir = os.path.join(fragment, 'imu')

        # 处理 gps
        if os.path.exists(gps_dir):
            for agent in ['front', 'back']:
                agent_dir = os.path.join(gps_dir, agent)
                if not os.path.isdir(agent_dir):
                    continue
                gps_files = glob.glob(os.path.join(agent_dir, '*.csv'))
                if not gps_files:
                    continue
                gps_file = gps_files[0]
                df = pd.read_csv(gps_file)

                agent_new = '2' if agent == 'front' else '1'

                for _, row in df.iterrows():
                    time = str(int(row['Time']))
                    yaml_data = {
                        'ego_speed': row['velocity'],
                        'lidar_pose': [
                            row['Latitude'],
                            row['Longitude'],
                            0.0,
                            0.0,
                            row['heading'],
                            0.0
                        ]
                    }
                    save_dir = os.path.join(new_scene_dir, agent_new)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{time}.yaml")
                    yaml_safe_dump(yaml_data, save_path)

        # 处理 imu（无人机 top）
        if os.path.exists(imu_dir):
            top_dir = os.path.join(imu_dir, 'top')
            if os.path.isdir(top_dir):
                imu_files = glob.glob(os.path.join(top_dir, '*.csv'))
                if imu_files:
                    imu_file = imu_files[0]
                    df = pd.read_csv(imu_file)

                    agent_new = '3'

                    for _, row in df.iterrows():
                        if pd.isnull(row['timestamp']):
                            continue
                        time = str(int(row['timestamp']))
                        # 这里你如果有无人机速度可以替换0.0
                        yaml_data = {
                            'ego_speed': 0.0,
                            'lidar_pose': [
                                row['linear_acceleration_x'],
                                row['linear_acceleration_y'],
                                row['linear_acceleration_z'],
                                row['angular_velocity_x'],
                                row['angular_velocity_y'],
                                row['angular_velocity_z']
                            ]
                        }
                        save_dir = os.path.join(new_scene_dir, agent_new)
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{time}.yaml")
                        yaml_safe_dump(yaml_data, save_path)

def main():
    old_dataset_root = '/home/beikh/workspace/xtreme1/datasets/datasets'
    new_dataset_root = '/data/datasets/TriCo3D/train'

    # 获取场景列表
    scene_dirs = glob.glob(os.path.join(old_dataset_root, '[0-9][0-9]'))
    tasks = []
    for scene_dir in scene_dirs:
        scene_id = os.path.basename(scene_dir)
        new_scene_dir = os.path.join(new_dataset_root, scene_id)
        tasks.append((scene_dir, new_scene_dir))

    with Pool(8) as p:
        p.map(process_scene, tasks)

if __name__ == '__main__':
    main()
