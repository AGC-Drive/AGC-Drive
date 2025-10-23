import os
import json
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

# 计算最终的转换矩阵
def compute_final_rt(fragment_path, timestamp, is_car=True):
    t_sec = int(os.path.basename(fragment_path))
    gps_front_file = os.path.join(fragment_path, 'gps', 'front', f"{t_sec}.csv")
    gpsA = read_gps_csv(gps_front_file, timestamp)
    if gpsA is None:
        print(f"前车 GPS 数据缺失 {timestamp}")
        return None

    RT_init = None
    if is_car:
        gps_back_file = os.path.join(fragment_path, 'gps', 'back', f"{t_sec}.csv")
        gpsB = read_gps_csv(gps_back_file, timestamp)
        if gpsB is None:
            print(f"后车 GPS 数据缺失 {timestamp}")
            return None
        RT_init = compute_init_rt(gpsA, gpsB, is_car=True)
    else:
        imu_file = os.path.join(fragment_path, 'imu', 'top', f"{t_sec}.csv")
        imu_data = read_imu_csv(imu_file, timestamp)
        if imu_data is None:
            print(f"无人机 IMU 数据缺失 {timestamp}")
            return None
        RT_init = compute_init_rt(gpsA, imu=imu_data, is_car=False)

    # 读取 ICP 计算的矩阵
    icp_path = os.path.join(fragment_path, 'front', 'back2front', f"{timestamp}.npy" if is_car else f"{timestamp}_uav.npy")
    if not os.path.exists(icp_path):
        print(f"ICP 结果缺失 {icp_path}")
        return None

    RT_icp = np.load(icp_path)

    # 计算最终的转换矩阵
    RT_init_inv = np.eye(4)
    R = RT_init[:3, :3]
    t = RT_init[:3, 3].reshape(3, 1)
    RT_init_inv[:3, :3] = R.T
    RT_init_inv[:3, 3] = (-R.T @ t).flatten()

    RT_total = RT_icp @ RT_init_inv
    return RT_total.tolist()

# 读取 GPS 数据
def read_gps_csv(csv_path, target_time_int):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['Time']) == target_time_int:
                return [
                    float(row['Latitude']),
                    float(row['Longitude']),
                    0.0,  # 高度暂时设为 0.0
                    float(row['heading'])
                ]
    return None

# 读取 IMU 数据
def read_imu_csv(csv_path, target_time_int):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['timestamp']) == target_time_int:
                return (
                    float(row['orientation_x']),
                    float(row['orientation_y']),
                    float(row['orientation_z']),
                    float(row['orientation_w'])
                )
    return None

# 计算初始化转换矩阵
def compute_init_rt(gpsA, gpsB=None, imu=None, is_car=True):
    if is_car:
        return CarB2CarA(gpsA, gpsB)
    else:
        return uav2car(gpsA[3], *imu)

# 处理一个场景的逻辑
def process_scene(original_dataset_path, new_dataset_path, scene_id):
    original_scene_dir = os.path.join(original_dataset_path, scene_id)
    new_cooperative_dir = os.path.join(new_dataset_path, scene_id, 'cooperative')
    if not os.path.exists(new_cooperative_dir):
        print(f"[{scene_id}] cooperative 不存在，跳过")
        return

    json_files = [f for f in os.listdir(new_cooperative_dir) if f.endswith('.json')]
    if not json_files:
        print(f"[{scene_id}] 没有 json 文件，跳过")
        return

    for json_file in tqdm(json_files, desc=f"处理 {scene_id}"):
        timestamp = os.path.splitext(json_file)[0]

        fragment_dirs = [d for d in os.listdir(original_scene_dir)
                         if os.path.isdir(os.path.join(original_scene_dir, d)) and d != 'cooperative']

        final_back2front = None
        final_uav2car = None

        # 找到对应 _final.npy 文件并计算转换矩阵
        for fragment in fragment_dirs:
            front_dir = os.path.join(original_scene_dir, fragment, 'front')

            # 计算后车->前车矩阵
            back2front_file = os.path.join(front_dir, 'back2front', f"{timestamp}_final.npy")
            if os.path.exists(back2front_file):
                final_back2front = compute_final_rt(fragment, timestamp, is_car=True)

            # 计算无人机->前车矩阵
            uav2car_file = os.path.join(front_dir, 'uav2car', f"{timestamp}_final.npy")
            if os.path.exists(uav2car_file):
                final_uav2car = compute_final_rt(fragment, timestamp, is_car=False)

        json_path = os.path.join(new_cooperative_dir, json_file)
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        # 覆盖 pairwise_t_matrix
        for item in data:
            if isinstance(item, dict):
                if 'pairwise_t_matrix1' in item and final_back2front is not None:
                    item['pairwise_t_matrix1'] = final_back2front
                if 'pairwise_t_matrix2' in item and final_uav2car is not None:
                    item['pairwise_t_matrix2'] = final_uav2car

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

def main():
    original_dataset_path = '/home/beikh/workspace/xtreme1/datasets/datasets'
    new_dataset_path = '/data/datasets/TriCo3D/test'

    available_scenes = sorted([f for f in os.listdir(original_dataset_path) if os.path.isdir(os.path.join(original_dataset_path, f))])

    print("原数据集中的可用场景：", ' '.join(available_scenes))
    selected = input("请输入要处理的场景编号（空格分隔，留空则全部处理）：").strip()

    if selected:
        scene_ids = selected.split()
    else:
        scene_ids = available_scenes

    print(f"将处理以下场景：{scene_ids}")

    # 多进程
    with mp.Pool(processes=min(16, mp.cpu_count())) as pool:
        for scene_id in scene_ids:
            pool.apply_async(process_scene, args=(original_dataset_path, new_dataset_path, scene_id))
        pool.close()
        pool.join()

    print("✅ 全部处理完成")

if __name__ == "__main__":
    main()
