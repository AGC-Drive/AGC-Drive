import os
import shutil
from tqdm import tqdm

# 配置路径
src_root = '/home/beikh/workspace/xtreme1/datasets/datasets/04'
dst_root = '/data/datasets/TriCo3D/train/04'

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

# 遍历所有片段
segments = os.listdir(src_root)
for segment in tqdm(segments, desc='Processing segments'):
    segment_path = os.path.join(src_root, segment)
    if not os.path.isdir(segment_path):
        continue

    # 遍历每个智能体：front, back, top
    for agent in os.listdir(segment_path):
        agent_path = os.path.join(segment_path, agent)
        if not os.path.isdir(agent_path):
            continue

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

        # 复制相机图片并重命名
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

print("✅ 数据整理完成！")
