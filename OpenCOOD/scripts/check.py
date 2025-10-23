import os
import json

# 文件夹路径
folder_path = "/data/datasets/AGC-Drive/train/05/cooperative"

# 遍历文件夹中的所有 JSON 文件
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                
                # 检查所有 item 中是否都没有 "objects" 键
                all_no_objects = True
                for item in data:
                    if "objects" in item:
                        all_no_objects = False
                        break  # 如果找到一个有 "objects" 的字典，跳出循环
                
                if all_no_objects:
                    print(f"文件 {filename} 中所有的 item 都没有 'objects' 键")
                # else:
                #     print(f"文件 {filename} 中至少有一个 item 包含 'objects' 键")
            
            except json.JSONDecodeError:
                print(f"文件 {filename} 无法解析为 JSON 格式")
print('done!')