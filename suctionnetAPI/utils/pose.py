__author__ = 'Minghao Gou'
__version__ = '1.0'
"""
定义Pose类以及与该类相关的函数。
"""

import numpy as np
from . import trans3d
from transforms3d.euler import euler2quat

class Pose:
    def __init__(self,id,x,y,z,alpha,beta,gamma):
        # 初始化位姿对象，包含物体id、平移（x, y, z）和欧拉角（alpha, beta, gamma，单位为度）
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        # alpha, beta, gamma为欧拉角，单位为度
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.quat = self.get_quat()           # 四元数表示
        self.mat_4x4 = self.get_mat_4x4()     # 4x4变换矩阵
        self.translation = self.get_translation() # 平移向量

    def __repr__(self):
        # 返回该位姿对象的字符串描述，便于打印和调试
        return '\nPose id=%d,x=%f,y=%f,z=%f,alpha=%f,beta=%f,gamma=%f' %(self.id,self.x,self.y,self.z,self.alpha,self.beta,self.gamma)+'\n'+'translation:'+self.translation.__repr__() + '\nquat:'+self.quat.__repr__()+'\nmat_4x4:'+self.mat_4x4.__repr__()

    def get_id(self):
        """
        功能:
        返回该对象的id编号
        """
        return self.id

    def get_translation(self):
        """ 
        功能:
        将self.x, self.y, self.z转换为平移向量self.translation
        返回值:
        numpy数组，形状为(3,)
        """
        return np.array([self.x,self.y,self.z])

    def get_quat(self):
        """
        功能:
        将self.alpha, self.beta, self.gamma（欧拉角，单位为度）转换为四元数self.quat
        返回值:
        numpy数组，四元数表示
        """
        euler = np.array([self.alpha, self.beta, self.gamma]) / 180.0 * np.pi
        quat = euler2quat(euler[0],euler[1],euler[2])
        return quat

    def get_mat_4x4(self):
        """
        功能:
        将self.x, self.y, self.z, self.alpha, self.beta, self.gamma转换为4x4位姿变换矩阵
        返回值:
        numpy数组，4x4变换矩阵
        """
        mat_4x4 = trans3d.get_mat(self.x,self.y,self.z,self.alpha,self.beta,self.gamma)
        return mat_4x4

def pose_from_pose_vector(pose_vector):
    """
    输入:
    pose_vector: 一个长度为7的列表，格式为[id,x,y,z,alpha,beta,gamma]
    
    输出:
    返回一个Pose类实例
    """
    return Pose(id = pose_vector[0],
    x = pose_vector[1],
    y = pose_vector[2],
    z = pose_vector[3],
    alpha = pose_vector[4],
    beta = pose_vector[5],
    gamma = pose_vector[6])

def pose_list_from_pose_vector_list(pose_vector_list):
    """
    输入:
    pose_vector_list: 由xmlhandler.py定义的位姿向量列表，每个元素为[id,x,y,z,alpha,beta,gamma]

    输出:
    返回一个Pose对象列表
    """
    pose_list = []
    for pose_vector in pose_vector_list:
        pose_list.append(pose_from_pose_vector(pose_vector))
    return pose_list