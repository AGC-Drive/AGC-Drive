import os
import json
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def process_scene(original_dataset_path, new_dataset_path, scene_id):
    original_scene_dir = os.path.join(original_dataset_path, scene_id)
    new_cooperative_dir = os.path.join(new_dataset_path, scene_id, 'cooperative')
    if not os.path.exists(new_cooperative_dir):
        print(f"[{scene_id}] cooperative 不存在，跳过")
        return

    fragment_dirs = [d for d in os.listdir(original_scene_dir)
                     if os.path.isdir(os.path.join(original_scene_dir, d)) and d != 'cooperative']

    for fragment in fragment_dirs:
        front_dir = os.path.join(original_scene_dir, fragment, 'front')

        back2front_dir = os.path.join(front_dir, 'back2front')
        uav2car_dir = os.path.join(front_dir, 'uav2car')

        # 全部 back2front npy
        if os.path.exists(back2front_dir):
            for npy_file in os.listdir(back2front_dir):
                if not npy_file.endswith('.npy'):
                    continue

                json_path = os.path.join(new_cooperative_dir, os.path.splitext(npy_file)[0] + '.json')
                if not os.path.exists(json_path):
                    continue

                with open(json_path, 'r') as f:
                    data = json.load(f)

                pairwise_t_matrix1 = np.load(os.path.join(back2front_dir, npy_file)).tolist()

                # 检查是否已有，防止重复插入
                keys = [list(item.keys())[0] for item in data if isinstance(item, dict)]
                if 'pairwise_t_matrix1' not in keys:
                    data.insert(0, {'pairwise_t_matrix1': pairwise_t_matrix1})

                    with open(json_path, 'w') as f:
                        json.dump(data, f, indent=4)

        # 全部 uav2car npy
        if os.path.exists(uav2car_dir):
            for npy_file in os.listdir(uav2car_dir):
                if not npy_file.endswith('.npy'):
                    continue

                json_path = os.path.join(new_cooperative_dir, os.path.splitext(npy_file)[0] + '.json')
                if not os.path.exists(json_path):
                    continue

                with open(json_path, 'r') as f:
                    data = json.load(f)

                pairwise_t_matrix2 = np.load(os.path.join(uav2car_dir, npy_file)).tolist()

                # 检查是否已有，防止重复插入
                keys = [list(item.keys())[0] for item in data if isinstance(item, dict)]
                if 'pairwise_t_matrix2' not in keys:
                    insert_index = 1 if 'pairwise_t_matrix1' in keys else 0
                    data.insert(insert_index, {'pairwise_t_matrix2': pairwise_t_matrix2})

                    with open(json_path, 'w') as f:
                        json.dump(data, f, indent=4)

def main():
    original_dataset_path = '/home/beikh/workspace/xtreme1/datasets/datasets'
    new_dataset_path = '/data/datasets/AGC-Drive/train'

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
