import os
import shutil
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 相机映射
camera_mapping = {
    'front1': 'camera1',
    'front2': 'camera2',
    'left': 'camera3',
    'right': 'camera4',
    'back': 'camera5'
}

# 确保目标目录存在
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 处理单个 agent 的复制任务
def process_agent(src_root, dst_root, segment, agent, top_record):
    segment_path = os.path.join(src_root, segment)
    agent_path = os.path.join(segment_path, agent)
    if not os.path.isdir(agent_path):
        return

    dst_agent_path = os.path.join(dst_root, agent)
    ensure_dir(dst_agent_path)

    # 复制 lidar 文件
    lidar_path = os.path.join(agent_path, 'lidar')
    if os.path.exists(lidar_path):
        for file in os.listdir(lidar_path):
            if file.endswith('.pcd'):
                src_file = os.path.join(lidar_path, file)
                dst_file = os.path.join(dst_agent_path, file)
                shutil.copy2(src_file, dst_file)

                # 如果是top，记录lidar文件名（不带后缀）
                if agent == 'top':
                    file_id = os.path.splitext(file)[0]
                    top_record.append(file_id)

    # 复制相机图片
    if agent == 'top':
        # 直接复制 camera 目录下的图片
        camera_path = os.path.join(agent_path, 'camera')
        if os.path.exists(camera_path):
            for file in os.listdir(camera_path):
                if file.endswith('.png'):
                    src_file = os.path.join(camera_path, file)
                    dst_file = os.path.join(dst_agent_path, file)
                    shutil.copy2(src_file, dst_file)
    else:
        # 普通agent，按映射表重命名
        for cam_folder, cam_name in camera_mapping.items():
            cam_path = os.path.join(agent_path, cam_folder)
            if os.path.exists(cam_path):
                for file in os.listdir(cam_path):
                    if file.endswith('.png'):
                        frame_id = os.path.splitext(file)[0]
                        new_name = f"{frame_id}_{cam_name}.png"
                        src_file = os.path.join(cam_path, file)
                        dst_file = os.path.join(dst_agent_path, new_name)
                        shutil.copy2(src_file, dst_file)

# 处理单个 segment 下所有 agent
def process_segment(src_root, dst_root, segment, top_record):
    segment_path = os.path.join(src_root, segment)
    if not os.path.isdir(segment_path):
        return 0

    agents = os.listdir(segment_path)
    for agent in agents:
        agent_path = os.path.join(segment_path, agent)
        if not os.path.isdir(agent_path):
            continue
        process_agent(src_root, dst_root, segment, agent, top_record)
    return 1

# 主程序
if __name__ == '__main__':
    src_base_root = '/home/beikh/workspace/xtreme1/datasets/datasets'
    dst_base_root = '/data/datasets/TriCo3D/train'

    dataset_ids = [f"{i:02d}" for i in range(14)]

    tasks = []
    top_records = {ds_id: [] for ds_id in dataset_ids}  # 存放每个数据集的top lidar文件名

    with ThreadPoolExecutor(max_workers=16) as executor:
        for ds_id in dataset_ids:
            src_root = os.path.join(src_base_root, ds_id)
            dst_root = os.path.join(dst_base_root, ds_id)
            if not os.path.exists(src_root):
                continue

            ensure_dir(dst_root)
            segments = os.listdir(src_root)
            for segment in segments:
                future = executor.submit(process_segment, src_root, dst_root, segment, top_records[ds_id])
                tasks.append(future)

        for _ in tqdm(as_completed(tasks), total=len(tasks), desc="Processing all datasets"):
            pass

    # 写入统一的 TriCo3D-V2U.json
    json_path = os.path.join(dst_base_root, 'TriCo3D-V2U.json')
    with open(json_path, 'w') as f:
        json.dump(top_records, f, indent=4)

    print("✅ 全部数据整理完成！")
