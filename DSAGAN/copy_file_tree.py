import os
import shutil

def copy_folder_structure(src_folder, dest_folder):
    # 遍历源文件夹
    for root, dirs, files in os.walk(src_folder):
        # 构建目标文件夹路径
        dest_root = root.replace(src_folder, dest_folder, 1)

        # 创建目标文件夹
        if not os.path.exists(dest_root):
            os.makedirs(dest_root)

        # 遍历文件
        for file in files:
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_root, file)

            # 复制文件（如果不是图片文件）
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                shutil.copy2(src_file_path, dest_file_path)

# 源文件夹路径
src_folder = '/home/k611/data3/lhy/triplanenet/triplanenet-main/data_lhy/selected_lhy2_croped'
# 目标文件夹路径
dest_folder = '/home/k611/data3/lhy/triplanenet/triplanenet-main/test/selected_lhy2_croped'

# 调用函数复制文件夹结构
copy_folder_structure(src_folder, dest_folder)