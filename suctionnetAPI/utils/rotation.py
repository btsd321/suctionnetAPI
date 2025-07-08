import numpy as np
from math import pi

def rotation_matrix(alpha, beta, gamma):
    # 根据欧拉角alpha、beta、gamma（弧度）生成旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    R = Rz.dot(Ry).dot(Rx)
    return R

def matrix_to_dexnet_params(matrix):
    # 将旋转矩阵转换为Dex-Net抓取参数（binormal和旋转角度）
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
        angle = pi * 2 - angle
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
    return matrix.astype(np.float32)

def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):
    # 批量将朝向向量和旋转角度转换为旋转矩阵
    axis_x = batch_towards
    ones = np.ones(axis_x.shape[0], dtype=axis_x.dtype)
    zeros = np.zeros(axis_x.shape[0], dtype=axis_x.dtype)
    axis_y = np.stack([-axis_x[:,1], axis_x[:,0], zeros], axis=-1)
    mask_y = (np.linalg.norm(axis_y, axis=-1) == 0)
    axis_y[mask_y] = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x, axis=-1, keepdims=True)
    axis_y = axis_y / np.linalg.norm(axis_y, axis=-1, keepdims=True)
    axis_z = np.cross(axis_x, axis_y)
    sin = np.sin(batch_angle)
    cos = np.cos(batch_angle)
    # 构造每个角度的绕x轴旋转矩阵
    R1 = np.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], axis=-1)
    R1 = R1.reshape([-1,3,3])
    # 构造每个朝向的基坐标系
    R2 = np.stack([axis_x, axis_y, axis_z], axis=-1)
    matrix = np.matmul(R2, R1)
    return matrix.astype(np.float32)

def viewpoint_to_matrix(towards):
    # 根据朝向向量生成正交基旋转矩阵
    axis_x = towards
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2
    return matrix
    
def batch_viewpoint_to_matrix(batch_towards):
    # 批量将朝向向量转换为正交基旋转矩阵
    axis_x = batch_towards
    zeros = np.zeros(axis_x.shape[0], dtype=axis_x.dtype)
    axis_y = np.stack([-axis_x[:,1], axis_x[:,0], zeros], axis=-1)
    mask_y = (np.linalg.norm(axis_y, axis=-1) == 0)
    axis_y[mask_y] = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x, axis=-1, keepdims=True)
    axis_y = axis_y / np.linalg.norm(axis_y, axis=-1, keepdims=True)
    axis_z = np.cross(axis_x, axis_y)
    R = np.stack([axis_x, axis_y, axis_z], axis=-1)
    # matrix = np.matmul(R2, R1)
    return R.astype(np.float32)

def dexnet_params_to_matrix(binormal, angle):
    # 根据Dex-Net参数（binormal和角度）生成旋转矩阵
    axis_y = binormal
    axis_x = np.array([axis_y[1], -axis_y[0], 0])
    if np.linalg.norm(axis_x) == 0:
        axis_x = np.array([1, 0, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    # 绕y轴旋转
    R1 = np.array([[np.cos(angle), 0, np.sin(angle)],
                  [0, 1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]])
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2.dot(R1)
    return matrix