import numpy as np
import open3d as o3d
import point_cloud_utils as pcu
import os


def create_dense_point_cloud(model_root, save_root):
    '''
    输入参数:
    - model_root: 字符串，模型文件夹的根目录
    - save_root: 字符串，保存生成的稠密点云的目录

    输出:
    - 无返回值，函数会将生成的稠密点云保存到save_root目录下
    '''

    models = []
    for i in range(88):
        models.append('%03d' % i)  # 生成模型编号列表，如'000', '001', ..., '087'

    for model in models:
        mesh_dir = os.path.join(model_root, model, 'textured.obj')  # 构建当前模型的mesh文件路径
        save_dir = os.path.join(save_root, model)  # 构建当前模型的保存目录
        os.makedirs(save_dir, exist_ok=True)  # 若目录不存在则创建

        print('Read mesh from:', mesh_dir)
        mesh = o3d.io.read_triangle_mesh(mesh_dir)  # 读取三角网格模型
        v = np.asarray(mesh.vertices)  # 获取顶点坐标
        f = np.asarray(mesh.triangles)  # 获取三角面片索引
        n = np.asarray(mesh.vertex_normals)  # 获取顶点法向量

        # 使用泊松盘采样算法对mesh进行稠密采样，num_samples=-1表示自动确定采样点数，radius为采样半径
        v_poisson, n_poisson = pcu.sample_mesh_poisson_disk(
            v, f, n, num_samples=-1, radius=0.0002, use_geodesic_distance=True)

        save_file = os.path.join(save_dir, model+'.npz')  # 构建保存文件名
        # 保存采样得到的点和法向量到npz文件
        np.savez(save_file, points=v_poisson, normals=n_poisson)

