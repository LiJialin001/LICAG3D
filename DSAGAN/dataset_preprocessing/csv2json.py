import csv
import json
import numpy as np


def dicom_to_world_coordinates(result, file_path):
    data = []
    
    for key, inner_dict in result.items():
        # print(key)
        primary_angle = inner_dict.get('PositionerPrimaryAngle')
        if not primary_angle:
            raise ValueError("Positioner Primary Angle information is missing in DICOM file.")
        secondary_angle = inner_dict.get('PositionerSecondaryAngle')
        if not secondary_angle:
            raise ValueError("Positioner Secondary Angle information is missing in DICOM file.")
        DSD = inner_dict.get('DistanceSourceToDetector')
        if not DSD:
            raise ValueError("Distance Source To Detector information is missing in DICOM file.")
        DSP = inner_dict.get('DistanceSourceToPatient')
        if not DSP:
            raise ValueError("Distance Source To Patient information is missing in DICOM file.")    
        ERMF = inner_dict.get('EstimatedRadiographicMagnificationFactor')
        if not ERMF:
            raise ValueError("Estimated Radiographic Magnification Factor information is missing in DICOM file.")
        IPS = inner_dict.get('IPS0')
        if not IPS:
            raise ValueError("IPS0 information is missing in DICOM file.")
        IAFADP = inner_dict.get('ImageAndFluoroscopyAreaDoseProduct')
        if not IAFADP:
            raise ValueError("ImageAndFluoroscopyAreaDoseProduct information is missing in DICOM file.")  
        resolution = inner_dict.get('resolution')
        if not resolution:
            raise ValueError("resolution information is missing in DICOM file.")         
        
        # 根据需要的参数进行坐标变换
        cam2world_matrix = campolar2rotation(primary_angle, secondary_angle, DSP)
        
        fx = float(DSD) / (float(IPS)*float(resolution))
        
        
        cam2world_seq = cam2world_matrix.reshape(1, 16).tolist()
        cam2world_seq[0] = cam2world_seq[0] + [fx, 0, 0.5, 0, fx, 0.5, 0, 0, 1]
        
        new_data = [
            key, cam2world_seq[0]
        ]
        
        # 使用split()方法将字符串分割成列表
        split_list = key.split('_')

        # 获取第二个下划线后的数字
        # number = split_list[3]
        # print(number)
        # if number == '3' :
        #     print('x')
                    
        data.append(new_data)
    
    with open(file_path, 'w') as json_file:
        json.dump({'labels': data}, json_file)
        

def campolar2rotation(a, b, r):
    r = float(r)/ 157.7
    theta = np.deg2rad(float(b)-90)  # 极角
    phi = np.deg2rad(float(a))  # 极坐标

    x_camera = r * np.sin(theta) * np.cos(phi)
    y_camera = r * np.sin(theta) * np.sin(phi)
    z_camera = r * np.cos(theta)

    # 相机坐标系的三个轴在世界坐标系中的方向
    z_xc, z_yc, z_zc = -x_camera, -y_camera, -z_camera
    ###########################
    # x轴 平行yox平面
    x_xc, x_yc, x_zc = -1 / (x_camera + 10e-6), 1 / (y_camera + 10e-6), 0
    y_xc, y_yc, y_zc =((z_yc * x_zc - x_yc * z_zc), -(z_xc * x_zc - z_zc * x_xc), (z_xc * x_yc - z_yc * x_xc))
    ##################################
    #
    if y_zc > 0:
        x_xc, x_yc, x_zc = -x_xc, -x_yc, -x_zc
        y_xc, y_yc, y_zc = -y_xc, -y_yc, -y_zc

    # 计算方向矩阵 D
    D = np.array([[x_xc, y_xc, z_xc],
                  [x_yc, y_yc, z_yc],
                  [x_zc, y_zc, z_zc]])

    # 单位化 D 的列向量
    D_prime = D / (np.linalg.norm(D, axis=0)+10e-6)

    # 计算旋转矩阵 R
    R = D_prime
    # 计算平移矩阵 T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x_camera, y_camera, z_camera]
    return T

def compute_cam2world_matrix(primary_angle, secondary_angle, distance_to_patient):
    distance_to_patient = float(distance_to_patient)
    # 将角度转换为弧度
    primary_angle_rad = np.radians(float(primary_angle))
    secondary_angle_rad = np.radians(float(secondary_angle))

    # 计算相机的Z轴方向
    z = -np.sin(primary_angle_rad)

    # 计算相机的X轴方向
    x = np.sin(secondary_angle_rad) * np.cos(primary_angle_rad)

    # 计算相机的Y轴方向
    y = np.sin(secondary_angle_rad) * np.sin(primary_angle_rad)

    # 构建相机的平移向量
    translation = np.array([distance_to_patient * x, distance_to_patient * y, distance_to_patient * z])

    # 计算相机的Cam2World矩阵
    cam2world_matrix = np.array([
        [x, y, z, translation[0]],
        [-y, x, 0, translation[1]],
        [0, 0, 1, translation[2]],
        [0, 0, 0, 1]
    ])

    return cam2world_matrix

# # 示例输入
# primary_angle = 30.0  # PositionerPrimaryAngle，单位：度
# secondary_angle = 45.0  # PositionerSecondaryAngle，单位：度
# distance_to_patient = 100.0  # DistanceSourceToPatient，单位：mm

# # 计算相机的Cam2World矩阵
# cam2world = compute_cam2world_matrix(primary_angle, secondary_angle, distance_to_patient)

# print("Cam2World Matrix:")
# print(cam2world)

def csv2dict(file_path):
    data = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            inner_dict = {}
            for key, value in row.items():
                if key != '':
                    inner_dict[key] = value
            data[row['id']] = inner_dict
    return data

csv_path = "/home/star/Projects/ljl/DSADataset/hangyi_pic_attribute_R.csv"  # 替换为你的CSV文件路径
json_path = '/home/star/Projects/ljl/DSADataset/dataset.json'  # 保存JSON文件的路径
result = csv2dict(csv_path)
dicom_to_world_coordinates(result=result, file_path=json_path)

