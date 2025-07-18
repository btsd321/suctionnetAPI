from suctionnetAPI import SuctionNet

if __name__ == "__main__":
    dataset_root = '/DATA2/Benchmark/graspnet'
    camera = 'realsense'
    suctionnet = SuctionNet(root=dataset_root, camera=camera)

    # 检查数据集的完整性, 确保所有必要的数据文件都存在
    suctionnet.check_data_completeness()
    
    # 提供探索数据集的相关函数
    # 获取包含指定物体的所有场景ID列表
    object_ids = [0, 1, 2]  # 指定你感兴趣的物体ID
    scene_ids = suctionnet.getSceneIds(objIds=object_ids)

    # 获取指定场景中包含的所有物体ID列表
    scene_ids = [0, 1, 2]   # 指定你感兴趣的场景ID
    object_ids = suctionnet.getObjIds(scene_ids)

    # 还提供了从数据集中加载数据的相关函数
    # 获取物体模型(返回open3d.geometry.PointCloud格式的点云模型列表)
    object_ids = [0, 1, 2]  # 指定你感兴趣的物体ID
    model_list = suctionnet.loadObjModels(object_ids)

    # 获取物体模型(返回trimesh.Trimesh格式的三维网格模型列表)
    object_ids = [0, 1, 2]  # 指定你感兴趣的物体ID
    model_list = suctionnet.loadObjTrimesh(object_ids)

    # 获取指定物体的密封标签(返回字典, key为物体ID, value为标签)
    object_ids = [0, 1, 2]  # 指定你感兴趣的物体ID
    seal_labels = suctionnet.loadSealLabels(object_ids)

    # 获取指定场景的扭矩标签(返回字典, key为场景ID, value为标签)
    scene_ids = [0, 1, 2]   # 指定你感兴趣的场景ID
    wrench_labels = suctionnet.loadWrenchLabels(scene_ids)

    # 获取指定场景的碰撞标签(返回字典, key为场景ID, value为标签)
    scene_ids = [0, 1, 2]   # 指定你感兴趣的场景ID
    colli_labels = suctionnet.loadCollisionLabels(scene_ids)

    # 获取图像数据
    # 获取RGB格式的彩色图像
    rgb_img = suctionnet.loadRGB(sceneId=0, camera='realsense', annId=0)
    # 获取BGR格式的彩色图像
    bgr_img = suctionnet.loadBGR(sceneId=0, camera='realsense', annId=0)
    # 获取深度图像
    depth_img = suctionnet.loadDepth(sceneId=0, camera='realsense', annId=0)
    # 获取掩码图像
    mask_img = suctionnet.loadMask(sceneId=0, camera='realsense', annId=0)

    # 从深度图和彩色图中获取点云及其对应颜色
    # 返回格式可以是open3d或numpy
    points, colors = suctionnet.loadScenePointCloud(sceneId=0, camera='realsense', annId=0, align=False, format = 'open3d')

    # 获取场景的open3d点云模型列表
    model_list = suctionnet.loadSceneModel(sceneId=0, camera ='kinect', annId = 0, align = False)
