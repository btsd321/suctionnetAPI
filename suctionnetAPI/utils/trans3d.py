from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import quat2euler, euler2quat
import numpy as np

def get_pose(pose):
    # 从4x4位姿矩阵中提取平移和欧拉角(角度制)
    pos, quat = pose_4x4_to_pos_quat(pose)
    euler = np.array([quat2euler(quat)[0], quat2euler(quat)[1], quat2euler(quat)[2]])
    euler = euler * 180.0 / np.pi  # 弧度转角度
    alpha, beta, gamma = euler[0], euler[1], euler[2]
    x, y, z = pos[0], pos[1], pos[2]
    return x, y, z, alpha, beta, gamma

def get_mat(x, y, z, alpha, beta, gamma):
    """
    根据平移(x, y, z)和欧拉角(alpha, beta, gamma, 角度制)生成4x4位姿变换矩阵

    输入参数:
        x, y, z: 平移分量
        alpha, beta, gamma: 欧拉角(角度制)

    返回值:
        pose: 4x4的numpy数组, 表示齐次变换矩阵
    """
    try:
        euler = np.array([alpha, beta, gamma]) / 180.0 * np.pi  # 角度转弧度
        quat = np.array(euler2quat(euler[0], euler[1], euler[2]))  # 欧拉角转四元数
        pose = pos_quat_to_pose_4x4(np.array([x, y, z]), quat)  # 组装为4x4矩阵
        return pose
    except Exception as e:
        print(str(e))
        pass         

def pos_quat_to_pose_4x4(pos, quat):
    """
    将平移向量和四元数转换为4x4位姿矩阵

    输入参数:
        pos: 长度为3的位置向量
        quat: 长度为4的四元数

    返回值:
        pose: 4x4的numpy数组, 表示齐次变换矩阵
    """
    pose = np.zeros([4, 4])
    mat = quat2mat(quat)  # 四元数转旋转矩阵
    pose[0:3, 0:3] = mat[:, :]
    pose[0:3, -1] = pos[:]
    pose[-1, -1] = 1
    return pose

def pose_4x4_to_pos_quat(pose):
    """
    将4x4位姿矩阵分解为平移向量和四元数

    输入参数:
        pose: 4x4的numpy数组, 表示齐次变换矩阵

    返回值:
        pos: 长度为3的位置向量
        quat: 长度为4的四元数
    """
    mat = pose[:3, :3]  # 提取旋转部分
    quat = mat2quat(mat)  # 旋转矩阵转四元数
    pos = np.zeros([3])
    pos[0] = pose[0, 3]
    pos[1] = pose[1, 3]
    pos[2] = pose[2, 3]
    return pos,