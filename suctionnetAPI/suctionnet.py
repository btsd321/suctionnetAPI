__author__ = 'hwcao'
__version__ = '1.0'

# TODO
# check_data_completeness (wrench), showObjSuction, showSceneSuction, show6DPose, loadSuctionLabels, loadSuction

# SuctionNet-1Billion数据集的访问接口。
# 说明及部分代码修改自MSCOCO api

# SuctionNet是一个面向通用物体吸取抓取的开放项目, 持续丰富中。
# 当前发布了SuctionNet-1Billion, 这是一个大规模的通用物体吸取抓取基准数据集, 
# 也适用于其他相关领域(如6D位姿估计、未知物体分割等)。
# suctionnetapi是一个Python API, 辅助加载、解析和可视化SuctionNet中的标注信息。
# 更多关于SuctionNet的数据、论文和教程, 请访问 https://graspnet.net/。
# 标注的具体格式也在官网有详细说明。suctionnetapi_demo.ipynb中有API的使用示例。
# 除了本API外, 也可以直接将标注文件加载为Python字典。
# 使用API可获得更多实用功能。注意本API同时支持*吸取抓取*和*6D位姿*标注。
# 对于6D位姿, 部分函数未定义(如碰撞相关)。

# 本API定义了如下主要函数：
#  SuctionNet             - SuctionNet主类, 加载SuctionNet标注文件并准备数据结构。
#  getSceneIds            - 获取满足条件的场景ID列表。
#  getObjIds              - 获取满足条件的物体ID列表。
#  getDataIds             - 获取满足条件的数据ID列表。
#  loadSuctionLabels      - 加载指定物体ID的吸取标签。
#  loadObjModels          - 加载指定物体ID的三维模型。
#  loadCollisionLabels    - 加载指定场景ID的碰撞标签。
#  loadSuction            - 加载指定场景和标注ID的吸取标签。
#  loadData               - 加载指定数据ID的数据路径。
#  showObjSuction         - 可视化指定物体ID的吸取姿态。
#  showSceneCollision     - 可视化指定场景ID的碰撞标签。
#  showSceneWrench        - 可视化指定场景ID的抗扭矩标签。
#  show6DPose             - 可视化指定场景ID的6D位姿, 将物体模型投影到点云上。
# 在API中, "ann"=annotation(标注), "obj"=object(物体), "img"=image(图像)。

# SuctionNet工具箱, 版本1.0
# 数据、论文和教程请见：https://graspnet.net/
# 代码作者：Hanwen Cao, 2021年。
# 仅限非商业用途, 遵循CC4.0协议 [见 https://graspnet.net/about]

import os
import numpy as np
from tqdm import tqdm
import open3d as o3d
import cv2
import trimesh

from .suction import Suction, SuctionGroup
from .utils.utils import generate_scene_model, plot_sucker_collision, transform_points, parse_posevector, create_table_cloud, get_model_suctions, \
    plot_sucker
from .utils.xmlhandler import xmlReader
from .utils.rotation import viewpoint_to_matrix


TOTAL_SCENE_NUM = 190

def _isArrayLike(obj):
    # 判断对象是否为可迭代且有长度(如list、numpy数组等)
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class SuctionNet():
    '''
    suctionnetAPI主类。

    输入参数:
    - root: 数据集根目录(字符串)
    - camera: 相机类型(字符串), 可选"kinect"或"realsense"
    - split: 数据集划分(字符串), 可选"all"、"train"、"test"、"test_seen"、"test_similar"、"test_novel"
    '''

    def __init__(self, root, camera='kinect', split='all'):
        # 初始化SuctionNet对象, 加载数据路径和基本信息
        assert camera in ['kinect', 'realsense'], 'camera应为kinect或realsense'
        assert split in ['all', 'train', 'test', 'test_seen', 'test_similar', 'test_novel'], 'split应为all/train/test/test_seen/test_similar/test_novel'
        self.root = root
        self.camera = camera
        self.split = split
        self.collisionLabels = {}

        # 根据split选择场景ID范围
        if split == 'all':
            self.sceneIds = list(range(TOTAL_SCENE_NUM))
        elif split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))

        # 预加载所有数据路径
        self.rgbPath = []
        self.depthPath = []
        self.segLabelPath = []
        self.metaPath = []
        self.sceneName = []
        self.annId = []

        for i in tqdm(self.sceneIds, desc='Loading data path...'):
            for img_num in range(256):
                self.rgbPath.append(os.path.join(
                    root, 'scenes', 'scene_'+str(i).zfill(4), camera, 'rgb', str(img_num).zfill(4)+'.png'))
                self.depthPath.append(os.path.join(
                    root, 'scenes', 'scene_'+str(i).zfill(4), camera, 'depth', str(img_num).zfill(4)+'.png'))
                self.segLabelPath.append(os.path.join(
                    root, 'scenes', 'scene_'+str(i).zfill(4), camera, 'label', str(img_num).zfill(4)+'.png'))
                self.metaPath.append(os.path.join(
                    root, 'scenes', 'scene_'+str(i).zfill(4), camera, 'meta', str(img_num).zfill(4)+'.mat'))
                self.sceneName.append('scene_'+str(i).zfill(4))
                self.annId.append(img_num)

        # self.objIds = self.getObjIds(self.sceneIds)
        self.objIds = None

    def __len__(self):
        # 返回数据集中的数据数量
        return len(self.depthPath)

    def check_data_completeness(self):
        '''
        检查数据集文件是否完整。

        输出:
        - bool类型, True表示完整, False表示有缺失
        '''
        error_flag = False
        # 检查模型文件
        for obj_id in tqdm(range(88), 'Checking Models'):
            if not os.path.exists(os.path.join(self.root, 'models','%03d' % obj_id, 'nontextured.ply')):
                error_flag = True
                print('No nontextured.ply For Object {}'.format(obj_id))
            if not os.path.exists(os.path.join(self.root, 'models','%03d' % obj_id, 'textured.sdf')):
                error_flag = True
                print('No textured.sdf For Object {}'.format(obj_id))
            if not os.path.exists(os.path.join(self.root, 'models','%03d' % obj_id, 'textured.obj')):
                error_flag = True
                print('No textured.obj For Object {}'.format(obj_id))
        # 检查稠密点云
        for obj_id in tqdm(range(88), 'Checking Dense Point Clouds'):
            if not os.path.exists(os.path.join(self.root, 'dense_point_clouds', '%03d.npz' % obj_id)):
                error_flag = True
                print('No Dense Point Cloud For Object {}'.format(obj_id))
        # 检查密封标签
        for obj_id in tqdm(range(88), 'Checking Seal Labels'):
            if not os.path.exists(os.path.join(self.root, 'seal_label', '%03d_seal.npz' % obj_id)):
                error_flag = True
                print('No Seal Label For Object {}'.format(obj_id))
        # 检查抗扭矩标签
        for sceneId in tqdm(self.sceneIds, 'Checking Wrench Labels'):
            if not os.path.exists(os.path.join(self.root, 'wrench_label', '%04d_wrench.npz' % sceneId)):
                error_flag = True
                print('No Wrench Label For Scene {}'.format(sceneId))
        # 检查碰撞标签
        for sceneId in tqdm(self.sceneIds, 'Checking Collosion Labels'):
            if not os.path.exists(os.path.join(self.root, 'suction_collision_label', '%04d_collision.npz' % sceneId)):
                error_flag = True
                print('No Collision Labels For Scene {}'.format(sceneId))
        # 检查场景数据
        for sceneId in tqdm(self.sceneIds, 'Checking Scene Datas'):
            scene_dir = os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId)
            if not os.path.exists(os.path.join(scene_dir,'object_id_list.txt')):
                error_flag = True
                print('No Object Id List For Scene {}'.format(sceneId))
            if not os.path.exists(os.path.join(scene_dir,'rs_wrt_kn.npy')):
                error_flag = True
                print('No rs_wrt_kn.npy For Scene {}'.format(sceneId))
            for camera in [self.camera]:
                camera_dir = os.path.join(scene_dir, camera)
                if not os.path.exists(os.path.join(camera_dir,'cam0_wrt_table.npy')):
                    error_flag = True
                    print('No cam0_wrt_table.npy For Scene {}, Camera:{}'.format(sceneId, camera))
                if not os.path.exists(os.path.join(camera_dir,'camera_poses.npy')):
                    error_flag = True
                    print('No camera_poses.npy For Scene {}, Camera:{}'.format(sceneId, camera)) 
                if not os.path.exists(os.path.join(camera_dir,'camK.npy')):
                    error_flag = True
                    print('No camK.npy For Scene {}, Camera:{}'.format(sceneId, camera))   
                for annId in range(256):
                    if not os.path.exists(os.path.join(camera_dir,'rgb','%04d.png' % annId)):
                        error_flag = True
                        print('No RGB Image For Scene {}, Camera:{}, annotion:{}'.format(sceneId, camera, annId))
                    if not os.path.exists(os.path.join(camera_dir,'depth','%04d.png' % annId)):
                        error_flag = True
                        print('No Depth Image For Scene {}, Camera:{}, annotion:{}'.format(sceneId, camera, annId))
                    if not os.path.exists(os.path.join(camera_dir,'label','%04d.png' % annId)):
                        error_flag = True
                        print('No Mask Label image For Scene {}, Camera:{}, annotion:{}'.format(sceneId, camera, annId))
                    if not os.path.exists(os.path.join(camera_dir,'meta','%04d.mat' % annId)):
                        error_flag = True
                        print('No Meta Data For Scene {}, Camera:{}, annotion:{}'.format(sceneId, camera, annId))
                    if not os.path.exists(os.path.join(camera_dir,'annotations','%04d.xml' % annId)):
                        error_flag = True
                        print('No Annotations For Scene {}, Camera:{}, annotion:{}'.format(sceneId, camera, annId))

        return not error_flag

    def getSceneIds(self, objIds=None):
        '''
        获取包含指定物体ID的所有场景ID。

        输入参数:
        - objIds: int或int列表, 物体ID

        输出:
        - 包含所有指定物体的场景ID列表
        '''
        if objIds is None:
            return self.sceneIds
        assert _isArrayLike(objIds) or isinstance(objIds, int), 'objIds必须为整数或整数列表/numpy数组'
        objIds = objIds if _isArrayLike(objIds) else [objIds]
        sceneIds = []
        for i in self.sceneIds:
            f = open(os.path.join(self.root, 'scenes', 'scene_' + str(i).zfill(4), 'object_id_list.txt'))
            idxs = [int(line.strip()) for line in f.readlines()]
            check = all(item in idxs for item in objIds)
            if check:
                sceneIds.append(i)
        return sceneIds

    def getObjIds(self, sceneIds=None):
        '''
        获取指定场景ID中的所有物体ID。

        输入参数:
        - sceneIds: int或int列表, 场景ID

        输出:
        - 物体ID列表
        '''
        if sceneIds is None:
            return self.objIds
        assert _isArrayLike(sceneIds) or isinstance(sceneIds, int), 'sceneIds必须为整数或整数列表/numpy数组'
        sceneIds = sceneIds if _isArrayLike(sceneIds) else [sceneIds]
        objIds = []
        for i in sceneIds:
            f = open(os.path.join(self.root, 'scenes', 'scene_' + str(i).zfill(4), 'object_id_list.txt'))
            idxs = [int(line.strip()) for line in f.readlines()]
            objIds = list(set(objIds+idxs))
        return objIds

    def getDataIds(self, sceneIds=None):
        '''
        获取指定场景ID对应的数据ID。

        输入参数:
        - sceneIds: int或int列表, 场景ID

        输出:
        - 数据ID列表。可通过self.loadData(ids)访问数据。
        '''
        if sceneIds is None:
            return list(range(len(self.sceneName)))
        ids = []
        indexPosList = []
        for i in sceneIds:
            indexPosList += [ j for j in range(0,len(self.sceneName),256) if self.sceneName[j] == 'scene_'+str(i).zfill(4) ]
        for idx in indexPosList:
            ids += list(range(idx, idx+256))
        return ids

    def loadObjModels(self, objIds=None):
        '''
        加载指定物体ID的三维点云模型。

        输入参数:
        - objIds: int或int列表, 物体ID

        输出:
        - open3d.geometry.PointCloud对象列表
        '''
        objIds = self.objIds if objIds is None else objIds
        assert _isArrayLike(objIds) or isinstance(objIds, int), 'objIds必须为整数或整数列表/numpy数组'
        objIds = objIds if _isArrayLike(objIds) else [objIds]
        models = []
        for i in tqdm(objIds, desc='Loading objects...'):
            plyfile = os.path.join(self.root, 'models','%03d' % i, 'nontextured.ply')
            models.append(o3d.io.read_point_cloud(plyfile))
        return models

    def loadObjTrimesh(self, objIds=None):
        '''
        加载指定物体ID的三维Trimesh模型。

        输入参数:
        - objIds: int或int列表, 物体ID

        输出:
        - trimesh.Trimesh对象列表
        '''
        objIds = self.objIds if objIds is None else objIds
        assert _isArrayLike(objIds) or isinstance(objIds, int), 'objIds必须为整数或整数列表/numpy数组'
        objIds = objIds if _isArrayLike(objIds) else [objIds]
        models = []
        for i in tqdm(objIds, desc='Loading objects...'):
            plyfile = os.path.join(self.root, 'models','%03d' % i, 'nontextured.ply')
            models.append(trimesh.load(plyfile))
        return models

    def loadSealLabels(self, objIds=None):
        '''
        加载指定物体ID的密封标签。

        输入参数:
        - objIds: int或int列表, 物体ID

        输出:
        - 每个物体的密封标签字典(points, normals, scores)
        '''
        objIds = self.objIds if objIds is None else objIds
        assert _isArrayLike(objIds) or isinstance(objIds, int), 'objIds必须为整数或整数列表/numpy数组'
        objIds = objIds if _isArrayLike(objIds) else [objIds]
        graspLabels = {}
        for i in tqdm(objIds, desc='Loading seal labels...'):
            file = np.load(os.path.join(self.root, 'seal_label', '{}_seal.npz'.format(str(i).zfill(3))))
            graspLabels[i] = (file['points'].astype(np.float32), file['normals'].astype(np.float32), file['scores'].astype(np.float32))
        return graspLabels

    def loadWrenchLabels(self, sceneIds=None):
        '''
        加载指定场景ID的抗扭矩标签。

        输入参数:
        - sceneIds: int或int列表, 场景ID

        输出:
        - wrench标签字典
        '''
        sceneIds = self.sceneIds if sceneIds is None else sceneIds
        assert _isArrayLike(sceneIds) or isinstance(sceneIds, int), 'sceneIds必须为整数或整数列表/numpy数组'
        sceneIds = sceneIds if _isArrayLike(sceneIds) else [sceneIds]
        wrenchLabels = {}
        for sid in tqdm(sceneIds, desc='Loading wrench labels...'):
            labels = np.load(os.path.join(self.root, 'wrench_label', '%04d_wrench.npz' % sid))
            wrenchLabel = []
            for j in range(len(labels)):
                wrenchLabel.append(labels['arr_{}'.format(j)])
            wrenchLabels['scene_'+str(sid).zfill(4)] = wrenchLabel
        return wrenchLabels

    def loadCollisionLabels(self, sceneIds=None):
        '''
        加载指定场景ID的碰撞标签。

        输入参数:
        - sceneIds: int或int列表, 场景ID

        输出:
        - 碰撞标签字典
        '''
        sceneIds = self.sceneIds if sceneIds is None else sceneIds
        assert _isArrayLike(sceneIds) or isinstance(sceneIds, int), 'sceneIds必须为整数或整数列表/numpy数组'
        sceneIds = sceneIds if _isArrayLike(sceneIds) else [sceneIds]
        collisionLabels = {}
        for sid in tqdm(sceneIds, desc='Loading collision labels...'):
            labels = np.load(os.path.join(self.root, 'collision_label', '%04d_collision.npz' % sid))
            collisionLabel = []
            for j in range(len(labels)):
                collisionLabel.append(labels['arr_{}'.format(j)])
            collisionLabels['scene_'+str(sid).zfill(4)] = collisionLabel
        return collisionLabels

    def loadRGB(self, sceneId, camera, annId):
        '''
        加载指定场景、相机和标注ID的RGB图像(RGB顺序)。

        输入参数:
        - sceneId: 场景编号
        - camera: 相机类型
        - annId: 标注编号

        输出:
        - numpy数组, RGB顺序
        '''
        return cv2.cvtColor(cv2.imread(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'rgb', '%04d.png' % annId)), cv2.COLOR_BGR2RGB)

    def loadBGR(self, sceneId, camera, annId):
        '''
        加载指定场景、相机和标注ID的RGB图像(BGR顺序)。

        输入参数:
        - sceneId: 场景编号
        - camera: 相机类型
        - annId: 标注编号

        输出:
        - numpy数组, BGR顺序
        '''
        return cv2.imread(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'rgb', '%04d.png' % annId))

    def loadDepth(self, sceneId, camera, annId):
        '''
        加载指定场景、相机和标注ID的深度图像。

        输入参数:
        - sceneId: 场景编号
        - camera: 相机类型
        - annId: 标注编号

        输出:
        - numpy数组, 类型为np.uint16
        '''
        return cv2.imread(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'depth', '%04d.png' % annId), cv2.IMREAD_UNCHANGED)
 
    def loadMask(self, sceneId, camera, annId):
        '''
        加载指定场景、相机和标注ID的分割掩码。

        输入参数:
        - sceneId: 场景编号
        - camera: 相机类型
        - annId: 标注编号

        输出:
        - numpy数组, 类型为np.uint16
        '''
        return cv2.imread(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'label', '%04d.png' % annId), cv2.IMREAD_UNCHANGED)
   
    def loadWorkSpace(self, sceneId, camera, annId):
        '''
        获取当前帧的工作空间边界框(mask非零区域的边界)。

        输入参数:
        - sceneId: 场景编号
        - camera: 相机类型
        - annId: 标注编号

        输出:
        - (x1, y1, x2, y2)四元组, 表示工作空间的左上和右下角坐标
        '''
        mask = self.loadMask(sceneId, camera, annId)
        maskx = np.any(mask, axis=0)
        masky = np.any(mask, axis=1)
        x1 = np.argmax(maskx)
        y1 = np.argmax(masky)
        x2 = len(maskx) - np.argmax(maskx[::-1])
        y2 = len(masky) - np.argmax(masky[::-1]) 
        return (x1, y1, x2, y2)

    def loadScenePointCloud(self, sceneId, camera, annId, align=False, format = 'open3d'):
        '''
        加载指定场景、相机和标注ID的点云。

        输入参数:
        - sceneId: 场景编号
        - camera: 相机类型
        - annId: 标注编号
        - align: 是否对齐到桌面坐标系
        - format: 返回格式, 'open3d'或'numpy'

        输出:
        - open3d.geometry.PointCloud对象或(numpy点坐标, numpy颜色)元组
        '''
        colors = self.loadRGB(sceneId = sceneId, camera = camera, annId = annId).astype(np.float32) / 255.0
        depths = self.loadDepth(sceneId = sceneId, camera = camera, annId = annId)
        intrinsics = np.load(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'camK.npy'))
        fx, fy = intrinsics[0,0], intrinsics[1,1]
        cx, cy = intrinsics[0,2], intrinsics[1,2]
        s = 1000.0
        
        if align:
            camera_poses = np.load(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'camera_poses.npy'))
            camera_pose = camera_poses[annId]
            align_mat = np.load(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'cam0_wrt_table.npy'))
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
        if format == 'open3d':
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            return cloud
        elif format == 'numpy':
            return points, colors
        else:
            raise ValueError('Format必须为"open3d"或"numpy"。')

    def loadSceneModel(self, sceneId, camera = 'kinect', annId = 0, align = False):
        '''
        加载指定场景、相机和标注ID下的所有物体模型(点云)。

        输入参数:
        - sceneId: 场景编号
        - camera: 相机类型
        - annId: 标注编号
        - align: 是否对齐到桌面坐标系

        输出:
        - open3d.geometry.PointCloud对象列表
        '''
        if align:
            camera_poses = np.load(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'camera_poses.npy'))
            camera_pose = camera_poses[annId]
            align_mat = np.load(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'cam0_wrt_table.npy'))
            camera_pose = np.matmul(align_mat,camera_pose)
        scene_reader = xmlReader(os.path.join(self.root, 'scenes', 'scene_%04d' % sceneId, camera, 'annotations', '%04d.xml'% annId))
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
            plyfile = os.path.join(self.root, 'models', '%03d'%obj_idx, 'nontextured.ply')
            model = o3d.io.read_point_cloud(plyfile)
            points = np.array(model.points)
            if align:
                pose = np.dot(camera_pose, pose)
            points = transform_points(points, pose)
            model.points = o3d.utility.Vector3dVector(points)
            model_list.append(model)
            pose_list.append(pose)
        return model_list

    def loadData(self, ids=None, *extargs):
        '''
        加载指定数据ID的数据路径。

        输入参数:
        - ids: int或int列表, 数据ID
        - extargs: 额外参数。也可通过loadData(sceneId, camera, annId)方式调用

        输出:
        - 若ids为int, 返回单个数据路径元组
        - 若ids为None或列表, 返回数据路径列表元组
        '''
        if ids is None:
            return (self.rgbPath, self.depthPath, self.segLabelPath, self.metaPath, self.sceneName, self.annId)
        
        if len(extargs) == 0:
            if isinstance(ids, int):
                return (self.rgbPath[ids], self.depthPath[ids], self.segLabelPath[ids], self.metaPath[ids], self.sceneName[ids], self.annId[ids])
            else:
                return ([self.rgbPath[id] for id in ids],
                    [self.depthPath[id] for id in ids],
                    [self.segLabelPath[id] for id in ids],
                    [self.metaPath[id] for id in ids],
                    [self.sceneName[id] for id in ids],
                    [self.annId[id] for id in ids])
        if len(extargs) == 2:
            sceneId = ids
            camera, annId = extargs
            rgbPath = os.path.join(self.root, 'scenes', 'scene_'+str(sceneId).zfill(4), camera, 'rgb', str(annId).zfill(4)+'.png')
            depthPath = os.path.join(self.root, 'scenes', 'scene_'+str(sceneId).zfill(4), camera, 'depth', str(annId).zfill(4)+'.png')
            segLabelPath = os.path.join(self.root, 'scenes', 'scene_'+str(sceneId).zfill(4), camera, 'label', str(annId).zfill(4)+'.png')
            metaPath = os.path.join(self.root, 'scenes', 'scene_'+str(sceneId).zfill(4), camera, 'meta', str(annId).zfill(4)+'.mat')
            scene_name = 'scene_'+str(sceneId).zfill(4)
            return (rgbPath, depthPath, segLabelPath, metaPath, scene_name,annId)

    def showObjSuction(self, obj_id, visu_num):
        '''
        可视化指定物体的吸取点。

        输入参数:
        - obj_id: 物体ID
        - visu_num: 可视化的吸取点数量

        输出:
        - 无返回值, 弹出3D可视化窗口
        '''
        ply_dir = os.path.join(self.root, 'models', '%03d' % obj_id, 'nontextured.ply')
        model = o3d.io.read_point_cloud(ply_dir)
        
        radius = 0.01
        height = 0.1

        suckers = []
        
        seal_dir = os.path.join(self.root, 'seal_label')
        sampled_points, normals, scores, _ = get_model_suctions('%s/%03d_seal.npz'%(seal_dir, obj_id))

        point_inds = np.random.choice(sampled_points.shape[0], visu_num)
        np.random.shuffle(point_inds)
        
        sucker_params = []

        for point_ind in point_inds:
            target_point = sampled_points[point_ind]
            normal = normals[point_ind]
            score = scores[point_ind]

            R = viewpoint_to_matrix(normal)
            t = target_point

            sucker = plot_sucker(R, t, score, radius, height)
            suckers.append(sucker)
            sucker_params.append([target_point[0],target_point[1],target_point[2],normal[0],normal[1],normal[2],radius, height])
                
        o3d.visualization.draw_geometries([model, *suckers], width=1536, height=864)

    def showSceneCollision(self, scene_idx, anno_idx, camera, visu_num_each):
        '''
        可视化指定场景的碰撞标签。

        输入参数:
        - scene_idx: 场景编号
        - anno_idx: 标注编号
        - camera: 相机类型
        - visu_num_each: 每个物体可视化的吸取点数量

        输出:
        - 无返回值, 弹出3D可视化窗口
        '''
        scene_name = 'scene_%04d' % scene_idx
        model_list, obj_list, pose_list = generate_scene_model(self.root, scene_name, anno_idx, return_poses=True, camera=camera, align=True)
        table = create_table_cloud(1.0, 0.02, 1.0, dx=-0.5, dy=-0.5, dz=0, grid_size=0.01)

        camera_poses = np.load(os.path.join(self.root, 'scenes', scene_name, camera, 'camera_poses.npy'.format(camera)))
        camera_pose = camera_poses[anno_idx]
        table.points = o3d.utility.Vector3dVector(transform_points(np.asarray(table.points), camera_pose))
        
        collision_dir = os.path.join(self.root, 'suction_collision_label')
        collision_dump = np.load(os.path.join(collision_dir, '{:04d}_collision.npz'.format(scene_idx)))

        radius = 0.01
        height = 0.1

        num_obj = len(obj_list)
        
        for obj_i in range(len(obj_list)):
            suckers = []
            print('Checking ' + str(obj_i+1) + ' / ' + str(num_obj))
            obj_idx = obj_list[obj_i]
            trans = pose_list[obj_i]
            seal_dir = os.path.join(self.root, 'seal_label')
            sampled_points, normals, _, _ = get_model_suctions('%s/%03d_seal.npz'%(seal_dir, obj_idx))
            collisions = collision_dump['arr_{}'.format(obj_i)]

            point_inds = np.random.choice(sampled_points.shape[0], visu_num_each)
            np.random.shuffle(point_inds)
            
            sucker_params = []

            for point_ind in point_inds:
                target_point = sampled_points[point_ind]
                normal = normals[point_ind]
                # score = scores[point_ind]
                collision = collisions[point_ind]

                R = viewpoint_to_matrix(normal)
                t = transform_points(target_point[np.newaxis,:], trans).squeeze()
                R = np.dot(trans[:3,:3], R)
                sucker = plot_sucker_collision(R, t, collision, radius, height)
                suckers.append(sucker)
                sucker_params.append([target_point[0],target_point[1],target_point[2],normal[0],normal[1],normal[2],radius, height])
                
            o3d.visualization.draw_geometries([table, *model_list, *suckers], width=1536, height=864)

    def showSceneWrench(self, scene_idx, anno_idx, camera, visu_num_each):
        '''
        可视化指定场景的抗扭矩标签。

        输入参数:
        - scene_idx: 场景编号
        - anno_idx: 标注编号
        - camera: 相机类型
        - visu_num_each: 每个物体可视化的吸取点数量

        输出:
        - 无返回值, 弹出3D可视化窗口
        '''
        radius = 0.002
        height = 0.05

        scene_name = 'scene_%04d' % scene_idx
        model_list, obj_list, pose_list = generate_scene_model(self.root, scene_name, anno_idx, 
                                            return_poses=True, align=True, camera=camera)
        
        table = create_table_cloud(1.0, 0.01, 1.0, dx=-0.5, dy=-0.5, dz=0, grid_size=0.01)
        camera_poses = np.load(os.path.join(self.root, 'scenes', scene_name, camera, 'camera_poses.npy'))
        camera_pose = camera_poses[anno_idx]
        
        table.points = o3d.utility.Vector3dVector(transform_points(np.asarray(table.points), camera_pose))
        
        wrench_dir = os.path.join(self.root, 'wrench_label')
        wrench_dump = np.load(os.path.join(wrench_dir, '{:04d}_wrench.npz'.format(scene_idx)))
        num_obj = len(obj_list)
        
        seal_dir =  os.path.join(self.root, 'seal_label')
        for obj_i in range(len(obj_list)):
            print('Checking ' + str(obj_i+1) + ' / ' + str(num_obj))
            obj_idx = obj_list[obj_i]
            print('object id:', obj_idx)
            trans = pose_list[obj_i]

            sampled_points, normals, _, _ = get_model_suctions('%s/%03d_seal.npz'%(seal_dir, obj_idx))
            sampled_points = transform_points(sampled_points, trans)
            center = np.mean(sampled_points, axis=0)
            score = wrench_dump['arr_{}'.format(obj_i)]

            arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.015, 
                                                                cylinder_height=0.2, cone_height=0.04)
            arrow_points = np.asarray(arrow.vertices)
            arrow_points[:, 2] = -arrow_points[:, 2]
            arrow_points = arrow_points + center[np.newaxis,:]
            arrow.vertices = o3d.utility.Vector3dVector(arrow_points)
            
            point_inds = np.random.choice(sampled_points.shape[0], visu_num_each)
            np.random.shuffle(point_inds)
            suckers = []

            for point_ind in point_inds:
                target_point = sampled_points[point_ind]
                normal = normals[point_ind]
                
                R = viewpoint_to_matrix(normal)
                t = target_point
                R = np.dot(trans[:3,:3], R)
                sucker = plot_sucker(R, t, score[point_ind], radius, height)
                suckers.append(sucker)
                
            o3d.visualization.draw_geometries([table, *model_list, *suckers, arrow], width=1536, height=864)

    def show6DPose(self, scene_idx, anno_idx, camera):
        '''
        可视化指定场景的6D位姿(将物体模型投影到点云上)。

        输入参数:
        - scene_idx: 场景编号
        - anno_idx: 标注编号
        - camera: 相机类型

        输出:
        - 无返回值, 弹出3D可视化窗口
        '''
        scene_name = 'scene_%04d' % scene_idx
        model_list, _, _ = generate_scene_model(self.root, scene_name, anno_idx, return_poses=True, camera=camera, align=True)
        table = create_table_cloud(1.0, 0.02, 1.0, dx=-0.5, dy=-0.5, dz=0, grid_size=0.01)

        camera_poses = np.load(os.path.join(self.root, 'scenes', scene_name, camera, 'camera_poses.npy'.format(camera)))
        camera_pose = camera_poses[anno_idx]
        table.points = o3d.utility.Vector3dVector(transform_points(np.asarray(table.points), camera_pose))
        
        o3d.visualization.draw_geometries([table, *model_list], width=1536, height=864)

