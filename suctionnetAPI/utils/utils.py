import os
import open3d as o3d
import numpy as np
from PIL import Image
from transforms3d.euler import euler2mat

from .xmlhandler import xmlReader

class CameraInfo():
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        # 相机内参信息类
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def get_camera_intrinsic(camera):
    '''
    输入:
        camera: 字符串, 相机类型, "realsense" 或 "kinect"
    输出:
        intrinsic: numpy数组, 形状为(3, 3), 相机内参矩阵
    '''
    param = o3d.camera.PinholeCameraParameters()
    if camera == 'kinect':
        param.intrinsic.set_intrinsics(1280,720,631.55,631.21,638.43,366.50)
    elif camera == 'realsense':
        param.intrinsic.set_intrinsics(1280,720,927.17,927.37,651.32,349.62)
    intrinsic = param.intrinsic.intrinsic_matrix
    return intrinsic

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    # 根据深度图和相机内参生成点云
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def generate_views(N, phi=(np.sqrt(5)-1)/2, center=np.zeros(3, dtype=np.float32), R=1):
    # 生成N个均匀分布在球面上的视点
    idxs = np.arange(N, dtype=np.float32)
    Z = (2 * idxs + 1) / N - 1
    X = np.sqrt(1 - Z**2) * np.cos(2 * idxs * np.pi * phi)
    Y = np.sqrt(1 - Z**2) * np.sin(2 * idxs * np.pi * phi)
    views = np.stack([X,Y,Z], axis=1)
    views = R * np.array(views) + center
    return views

def generate_scene_model(dataset_root, scene_name, anno_idx, return_poses=False, align=False, camera='realsense'):
    # 生成场景中所有物体的点云模型(可选返回物体位姿)
    if align:
        print('align')
        camera_poses = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'camera_poses.npy'))
        camera_pose = camera_poses[anno_idx]
        align_mat = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'cam0_wrt_table.npy'))
        camera_pose = np.matmul(align_mat,camera_pose)
    print('Scene {}, {}'.format(scene_name, camera))
    scene_reader = xmlReader(os.path.join(dataset_root, 'scenes', scene_name, camera, 'annotations', '%04d.xml'%anno_idx))
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    mat_list = []
    model_list = []
    pose_list = []
    for posevector in posevectors:
        obj_idx, pose = parse_posevector(posevector)
        obj_list.append(obj_idx)
        mat_list.append(pose)

    for obj_idx, pose in zip(obj_list, mat_list):
        plyfile = os.path.join(dataset_root, 'models', '%03d'%obj_idx, 'nontextured.ply')
        model = o3d.io.read_point_cloud(plyfile)
        points = np.array(model.points)
        if align:
            pose = np.dot(camera_pose, pose)
        points = transform_points(points, pose)
        model.points = o3d.utility.Vector3dVector(points)
        model_list.append(model)
        pose_list.append(pose)

    if return_poses:
        return model_list, obj_list, pose_list
    else:
        return model_list

def generate_scene_pointcloud(dataset_root, scene_name, anno_idx, align=False, camera='kinect'):
    # 生成场景点云(包含颜色), 可选对齐到桌面坐标系
    colors = np.array(Image.open(os.path.join(dataset_root, 'scenes', scene_name, camera, 'rgb', '%04d.png'%anno_idx)), dtype=np.float32) / 255.0
    depths = np.array(Image.open(os.path.join(dataset_root, 'scenes', scene_name, camera, 'depth', '%04d.png'%anno_idx)))
    intrinsics = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'camK.npy'))
    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    s = 1000.0
    
    if align:
        camera_poses = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'camera_poses.npy'))
        camera_pose = camera_poses[anno_idx]
        align_mat = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'cam0_wrt_table.npy'))
        camera_pose = align_mat.dot(camera_pose)

    xmap, ymap = np.arange(colors.shape[1]), np.arange(colors.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask]
    colors = colors[mask]
    if align:
        points = transform_points(points, camera_pose)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    return cloud

def rotation_matrix(rx, ry, rz):
    # 根据欧拉角(弧度)生成旋转矩阵
    Rx = np.array([[1,          0,           0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [          0, 1,          0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [         0,           0, 1]])
    R = Rz.dot(Ry).dot(Rx)
    return R

def transform_matrix(tx, ty, tz, rx, ry, rz):
    # 根据平移和旋转参数生成4x4齐次变换矩阵
    trans = np.eye(4)
    trans[:3,3] = np.array([tx, ty, tz])
    rot_x = np.array([[1,          0,           0],
                      [0, np.cos(rx), -np.sin(rx)],
                      [0, np.sin(rx),  np.cos(rx)]])
    rot_y = np.array([[ np.cos(ry), 0, np.sin(ry)],
                      [          0, 1,          0],
                      [-np.sin(ry), 0, np.cos(ry)]])
    rot_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                      [np.sin(rz),  np.cos(rz), 0],
                      [         0,           0, 1]])
    trans[:3,:3] = rot_x.dot(rot_y).dot(rot_z)
    return trans

def matrix_to_dexnet_params(matrix):
    # 将旋转矩阵转换为Dex-Net抓取参数(binormal和旋转角度)
    approach = matrix[:, 0]
    binormal = matrix[:, 1]
    axis_y = binormal
    axis_x = np.array([axis_y[1], -axis_y[0], 0])
    if np.linalg.norm(axis_x) == 0:
        axis_x = np.array([1, 0, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R = np.c_[axis_x, np.c_[axis_y, axis_z]]
    approach = R.T.dot(approach)
    cos_t, sin_t = approach[0], -approach[2]
    angle = np.arccos(cos_t)
    if sin_t < 0:
        angle = np.pi * 2 - angle
    return binormal, angle

def viewpoint_params_to_matrix(towards, angle):
    # 根据朝向向量和旋转角度生成旋转矩阵
    axis_x = towards
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    if np.linalg.norm(axis_y) == 0:
        axis_y = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R1 = np.array([[1, 0, 0],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle), np.cos(angle)]])
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2.dot(R1)
    return matrix

def dexnet_params_to_matrix(binormal, angle):
    # 根据Dex-Net参数(binormal和角度)生成旋转矩阵
    axis_y = binormal
    axis_x = np.array([axis_y[1], -axis_y[0], 0])
    if np.linalg.norm(axis_x) == 0:
        axis_x = np.array([1, 0, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R1 = np.array([[np.cos(angle), 0, np.sin(angle)],
                  [0, 1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]])
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2.dot(R1)
    return matrix

def transform_points(points, trans):
    # 对点云进行齐次变换
    ones = np.ones([points.shape[0],1], dtype=points.dtype)
    points_ = np.concatenate([points, ones], axis=-1)
    points_ = np.matmul(trans, points_.T).T
    return points_[:,:3]

def get_model_suctions(datapath):
    # 读取模型吸取点、法向量、分数和碰撞信息
    dump = np.load(datapath)
    points = dump['points']
    normals = dump['normals']
    scores = dump['scores']
    collision = dump['collision']
    return points, normals, scores, collision

def parse_posevector(posevector):
    # 将位姿向量解析为4x4变换矩阵和物体编号
    mat = np.zeros([4,4],dtype=np.float32)
    alpha, beta, gamma = posevector[4:7]
    alpha = alpha / 180.0 * np.pi
    beta = beta / 180.0 * np.pi
    gamma = gamma / 180.0 * np.pi
    mat[:3,:3] = euler2mat(alpha, beta, gamma)
    mat[:3,3] = posevector[1:4]
    mat[3,3] = 1
    obj_idx = int(posevector[0])
    return obj_idx, mat

def create_mesh_cylinder(R, t, score, radius=0.01, height=0.1):
    # 创建吸盘圆柱体网格, 用于可视化吸取点
    cylinder = o3d.geometry.TriangleMesh().create_cylinder(radius, height)
    vertices = np.asarray(cylinder.vertices)[:, [2, 1, 0]]
    vertices[:, 0] += height / 2
    vertices = np.dot(R, vertices.T).T + t
    cylinder.vertices = o3d.utility.Vector3dVector(vertices)
    colors = np.array([score, 0, 0])
    colors = np.expand_dims(colors, axis=0)
    colors = np.repeat(colors, vertices.shape[0], axis=0)
    cylinder.vertex_colors = o3d.utility.Vector3dVector(colors)

    return cylinder

def create_mesh_cylinder_detection(R, t, collision, radius, height):
    # 创建带有碰撞信息的吸盘圆柱体网格
    cylinder = o3d.geometry.TriangleMesh().create_cylinder(radius, height)
    vertices = np.asarray(cylinder.vertices)[:, [2, 1, 0]]
    vertices[:, 0] += height / 2
    vertices = np.dot(R, vertices.T).T + t
    cylinder.vertices = o3d.utility.Vector3dVector(vertices)
    if collision:
        colors = np.array([0.7, 0, 0])
    else:
        colors = np.array([0, 0.7, 0])
    colors = np.expand_dims(colors, axis=0)
    colors = np.repeat(colors, vertices.shape[0], axis=0)
    cylinder.vertex_colors = o3d.utility.Vector3dVector(colors)

    return cylinder

def plot_sucker(R, t, score, radius=0.01, height=0.1):
    '''
        可视化吸盘
        center: 吸盘中心点
        R: 旋转矩阵
    '''
    return create_mesh_cylinder(R, t, score, radius, height)

def plot_sucker_collision(R, t, collision, radius=0.01, height=0.1):
    '''
        可视化带碰撞信息的吸盘
        center: 吸盘中心点
        R: 旋转矩阵
    '''
    return create_mesh_cylinder_detection(R, t, collision, radius, height)

def create_table_cloud(width, height, depth, dx=0, dy=0, dz=0, grid_size=0.01):
    # 创建桌面点云
    xmap = np.linspace(0, width, int(width/grid_size))
    ymap = np.linspace(0, depth, int(depth/grid_size))
    zmap = np.linspace(0, height, int(height/grid_size))
    xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
    xmap += dx
    ymap += dy
    zmap += dz
    points = np.stack([xmap, -ymap, -zmap], axis=-1)
    points = points.reshape([-1, 3])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud

def create_axis(length,grid_size = 0.01):
    # 创建三维坐标轴点云
    num = int(length / grid_size)
    xmap = np.linspace(0,length,num)
    ymap = np.linspace(0,2*length,num)
    zmap = np.linspace(0,3*length,num)
    x_p = np.vstack([xmap.T,np.zeros((1,num)),np.zeros((1,num))])
    y_p = np.vstack([np.zeros((1,num)),ymap.T,np.zeros((1,num))])
    z_p = np.vstack([np.zeros((1,num)),np.zeros((1,num)),zmap.T])
    p = np.hstack([x_p,y_p,z_p])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(p.T)
    return cloud

def plot_axis(R,center,length,grid_size = 0.01):
    # 可视化三维坐标轴
    num = int(length / grid_size)
    xmap = np.linspace(0,length,num)
    ymap = np.linspace(0,2*length,num)
    zmap = np.linspace(0,3*length,num)
    x_p = np.vstack([xmap.T,np.zeros((1,num)),np.zeros((1,num))])
    y_p = np.vstack([np.zeros((1,num)),ymap.T,np.zeros((1,num))])
    z_p = np.vstack([np.zeros((1,num)),np.zeros((1,num)),zmap.T])
    p = np.hstack([x_p,y_p,z_p])
    p = np.dot(R, p).T + center
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(p)
    return cloud

def find_scene_by_model_id(dataset_root, model_id_list):
    # 根据物体id列表查找包含这些物体的场景名称
    picked_scene_names = []
    scene_names = ['scene_'+str(i).zfill(4) for i in range(190)]
    for scene_name in scene_names:
        try:
            scene_reader = xmlReader(os.path.join(dataset_root, 'scenes', scene_name, 'kinect', 'annotations', '0000.xml'))
        except:
            continue
        posevectors = scene_reader.getposevectorlist()
        for posevector in posevectors:
            obj_idx, _ = parse_posevector(posevector)
            if obj_idx in model_id_list:
                picked_scene_names.append(scene_name)
                print(obj_idx, scene_name)
                break
    return picked_scene_names

def get_obj_pose_list(camera_pose, pose_vectors):
    # 根据相机位姿和物体位姿向量, 计算每个物体在相机坐标系下的4x4变换矩阵
    import numpy as np
    obj_list = []
    mat_list = []
    pose_list = []
    for posevector in pose_vectors:
        obj_idx, mat = parse_posevector(posevector)
        obj_list.append(obj_idx)
        mat_list.append(mat)

    for obj_idx, mat in zip(obj_list, mat_list):
        pose = np.dot(camera_pose, mat)
        pose_list.append(pose)

    return obj_list, pose_list

def batch_rgbdxyz_2_rgbxy_depth(points, camera):
    '''
    输入:
        points: np.array(-1,3), 相机坐标系下的点
        camera: 字符串, 相机类型
    输出:
        coords: float, 像素坐标xy, 形状为(-1,2)
        depths: float, 像素深度, 形状为(-1,)
    '''
    intrinsics = get_camera_intrinsic(camera)
    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    s = 1000.0
    depths = s * points[:,2] # point_z
    # 注意：x和y的计算顺序
    coords_x = points[:,0] / points[:,2] * fx + cx
    coords_y = points[:,1] / points[:,2] * fy + cy
    coords = np.stack([coords_x, coords_y], axis=-1)
    return coords, depths

def framexy_depth_2_xyz(pixel_x, pixel_y, depth, camera):
    '''
    输入:
        pixel_x: int, 像素x坐标
        pixel_y: int, 像素y坐标
        depth: float, 深度值, 单位为毫米
        camera: 字符串, 相机类型
    输出:
        x, y, z: float, 点在相机坐标系下的三维坐标, 单位为毫米
    '''
    intrinsics = get_camera_intrinsic(camera)
    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    z = depth # mm
    x = z / fx * (pixel_x - cx) # mm
    y = z / fy * (pixel_y - cy) # mm
    return x, y, z

def batch_framexy_depth_2_xyz(pixel_x, pixel_y, depth, camera):
    '''
    输入:
        pixel_x: numpy数组, 像素x坐标, 形状(-1,)
        pixel_y: numpy数组, 像素y坐标, 形状(-1,)
        depth: numpy数组, 深度值, 单位为毫米, 形状(-1,)
        camera: 字符串, 相机类型
    输出:
        x, y, z: numpy数组, 点在相机坐标系下的三维坐标, 单位为毫米
    '''
    intrinsics = get_camera_intrinsic(camera)
    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    z = depth # mm
    x = z / fx * (pixel_x - cx) # mm
    y = z / fy * (pixel_y - cy) # mm
    return x, y, z

def key_point_2_rotation(center_xyz, open_point_xyz, upper_point_xyz):
    '''
    输入:
        center_xyz: numpy数组, 中心点坐标
        open_point_xyz: numpy数组, 开口点坐标
        upper_point_xyz: numpy数组, 上方点坐标
    输出:
        rotation: numpy数组, 旋转矩阵
    '''
    open_point_vector = open_point_xyz - center_xyz
    upper_point_vector = upper_point_xyz - center_xyz
    unit_open_point_vector = open_point_vector / np.linalg.norm(open_point_vector)
    unit_upper_point_vector = upper_point_vector / np.linalg.norm(upper_point_vector)
    rotation = np.hstack((
        np.array([[0],[0],[1.0]]), 
        unit_open_point_vector.reshape((-1, 1)), 
        unit_upper_point_vector.reshape((-1, 1))
    ))
    return rotation

def batch_key_point_2_rotation(centers_xyz, open_points_xyz, upper_points_xyz):
    '''
    输入:
        centers_xyz: numpy数组, 中心点坐标, 形状(-1, 3)
        open_points_xyz: numpy数组, 开口点坐标, 形状(-1, 3)
        upper_points_xyz: numpy数组, 上方点坐标, 形状(-1, 3)
    输出:
        rotations: numpy数组, 旋转矩阵, 形状(-1, 3, 3)
    '''
    open_points_vector = open_points_xyz - centers_xyz # (-1, 3)
    upper_points_vector = upper_points_xyz - centers_xyz # (-1, 3)
    open_point_norm = np.linalg.norm(open_points_vector, axis = 1).reshape(-1, 1)
    upper_point_norm = np.linalg.norm(upper_points_vector, axis = 1).reshape(-1, 1)
    unit_open_points_vector = open_points_vector / np.hstack((open_point_norm, open_point_norm, open_point_norm)) # (-1, 3)
    unit_upper_points_vector = upper_points_vector / np.hstack((upper_point_norm, upper_point_norm, upper_point_norm)) # (-1, 3)
    num = open_points_vector.shape[0]
    x_axis = np.hstack((np.zeros((num, 1)), np.zeros((num, 1)), np.ones((num, 1)))).astype(np.float32).reshape(-1, 3, 1)
    rotations = np.dstack((x_axis, unit_open_points_vector.reshape((-1, 3, 1)), unit_upper_points_vector.reshape((-1, 3, 1))))
    return rotations

