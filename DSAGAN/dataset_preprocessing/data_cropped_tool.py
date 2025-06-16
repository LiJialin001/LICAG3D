import cv2
import os
from tqdm import tqdm

def process_image(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    
    # 边框剪裁
    image_height, image_width = image.shape[:2]
    if image_width >= 1024:
        border_size = 40
    elif image_width >= 512:
        border_size = 20
    else:
        # 图像宽度小于512，不进行剪裁
        border_size = 0
    cropped_image = image[border_size:image_height-border_size, border_size:image_width-border_size]
    
    resized_img = cv2.resize(cropped_image, (512, 512), interpolation=cv2.INTER_LINEAR)
    # resized_img = cropped_image
    
    # 如果图像是单通道，转换为三通道
    if len(resized_img.shape) == 2:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
    
    # 确定输出路径
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存处理后的图像
    cv2.imwrite(output_path, resized_img)

def process_images_in_folder(input_folder, output_folder):
    # 遍历输入文件夹下的所有子文件夹和图像文件
    for root, dirs, files in os.walk(input_folder):
        for file in tqdm(files, desc="Processing Images"):
            # 仅处理图像文件
            if file.endswith(".jpg") or file.endswith(".png"):
                # 构造输入和输出路径
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(image_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                
                # 处理图像并保存
                process_image(image_path, output_path)

# 输入文件夹路径
input_folder = "/home/l611/Projects3/ljl/eg3d/dataset_preprocessing/selected_lhy2"

# 输出文件夹路径
output_folder = "/home/l611/Projects3/ljl/eg3d/dataset_preprocessing/selected_lhy2_croped"

# 处理图像
process_images_in_folder(input_folder, output_folder)
