__author__ = 'hwcao'
__version__ = '1.0'

import numpy as np
# import open3d as o3d
import copy
# import cv2

from .utils.utils import plot_sucker

SUCTION_ARRAY_LEN = 8
EPS = 1e-8

class Suction():
    def __init__(self, *args):
        '''
        输入参数:
        - args 可以是一个numpy数组, 或(score, direction, translation, object_id)四元组
        - numpy数组格式为[score, direction(3), translation(3), object_id]
        - numpy数组长度为8
        '''
        if len(args) == 1:
            if type(args[0]) == np.ndarray:
                self.suction_array = copy.deepcopy(args[0])
            else:
                raise TypeError('如果只传入一个参数, 则必须为np.ndarray类型。')
        elif len(args) == 4:
            score, direction, translation, object_id = args
            self.suction_array = np.concatenate([np.array((score)),direction, translation, np.array((object_id)).reshape(-1)]).astype(np.float32)
        else:
            raise ValueError('只接受1个或4个参数')
    
    def __repr__(self):
        # 返回吸取点的字符串描述, 便于打印和调试
        return 'Suction: score:{}, translation:{}\ndirection:\n{}\nobject id:{}'.format(self.score(), self.translation(), self.direction(), self.object_id())

    def score(self):
        '''
        输出:
        - 吸取点的分数(float)
        '''
        return float(self.suction_array[0])

    def direction(self):
        '''
        输出:
        - 吸取点的方向, np.array, 形状为(3,), 表示旋转向量
        '''
        return self.suction_array[1:4]

    def translation(self):
        '''
        输出:
        - 吸取点的平移向量, np.array, 形状为(3,)
        '''
        return self.suction_array[4:7]

    def object_id(self):
        '''
        输出:
        - 吸取点所属物体的id(int)
        '''
        return int(self.suction_array[7])

    def to_open3d_geometry(self):
        '''
        输出:
        - 返回open3d几何体列表, 用于可视化吸盘
        '''
        return plot_sucker(R=self.direction(), t=self.translation(), score=self.score())

class SuctionGroup():
    def __init__(self, *args):
        '''
        输入参数:
        - 可不传参数, 或传入一个吸取点组的numpy数组
        '''
        if len(args) == 0:
            self.suction_group_array = np.zeros((0, SUCTION_ARRAY_LEN), dtype=np.float32)
        elif len(args) == 1:
            self.suction_group_array = args[0]
        else:
            raise ValueError('参数必须为空或为Suction实例的列表。')

    def __len__(self):
        '''
        输出:
        - 返回吸取点组的数量(int)
        '''
        return len(self.suction_group_array)

    def __repr__(self):
        # 返回吸取点组的字符串描述, 便于打印和调试
        repr = '----------\nSuction Group, Number={}:\n'.format(self.__len__())
        if self.__len__() <= 6:
            for suction_array in self.suction_group_array:
                repr += Suction(suction_array).__repr__() + '\n'
        else:
            for i in range(3):
                repr += Suction(self.suction_group_array[i]).__repr__() + '\n'
            repr += '......\n'
            for i in range(3):
                repr += Suction(self.suction_group_array[-(3-i)]).__repr__() + '\n'
        return repr + '----------'

    def __getitem__(self, index):
        '''
        输入:
        - index: int 或 slice类型
        输出:
        - 若index为int, 返回对应的Suction实例
        - 若index为slice, 返回对应的SuctionGroup实例
        '''
        if type(index) == int:
            return Suction(self.suction_group_array[index])
        elif type(index) == slice:
            suctiongroup = SuctionGroup()
            suctiongroup.suction_group_array = copy.deepcopy(self.suction_group_array[index])
            return suctiongroup
        else:
            raise TypeError('SuctionGroup的__getitem__只支持int或slice类型, 当前类型为{}'.format(type(index)))

    def scores(self):
        '''
        输出:
        - 返回所有吸取点的分数, numpy数组, 形状为(-1,)
        '''
        return self.suction_group_array[:,0]

    def directions(self):
        '''
        输出:
        - 返回所有吸取点的方向, numpy数组, 形状为(-1, 3)
        '''
        return self.suction_group_array[:, 1:4]

    def translations(self):
        '''
        输出:
        - 返回所有吸取点的平移向量, numpy数组, 形状为(-1, 3)
        '''
        return self.suction_group_array[:, 4:7]

    def object_ids(self):
        '''
        输出:
        - 返回所有吸取点所属物体的id, numpy数组, 形状为(-1,)
        '''
        return self.suction_group_array[:,7].astype(np.int32)

    def add(self, suction):
        '''
        输入:
        - suction: Suction实例
        功能:
        - 向吸取点组中添加一个吸取点
        '''
        self.suction_group_array = np.concatenate((self.suction_group_array, suction.suction_array.reshape((-1, SUCTION_ARRAY_LEN))))
        return self

    def remove(self, index):
        '''
        输入:
        - index: 要移除的吸取点索引列表
        功能:
        - 从吸取点组中移除指定索引的吸取点
        '''
        self.suction_group_array = np.delete(self.suction_group_array, index, axis = 0)
        return self

    def from_npy(self, npy_file_path):
        '''
        输入:
        - npy_file_path: 文件路径字符串
        功能:
        - 从npy或npz文件加载吸取点组
        '''
        if npy_file_path[-3:] == 'npz':
            self.suction_group_array = np.load(npy_file_path)['arr_0']
        else:
            self.suction_group_array = np.load(npy_file_path)
        return self

    def save_npy(self, npy_file_path):
        '''
        输入:
        - npy_file_path: 文件路径字符串
        功能:
        - 将吸取点组保存为npy文件
        '''
        np.save(npy_file_path, self.suction_group_array)

    def to_open3d_geometry_list(self):
        '''
        输出:
        - 返回所有吸取点的open3d几何体列表, 用于可视化
        '''
        geometry = []
        for i in range(len(self.suction_group_array)):
            g = Suction(self.suction_group_array[i])
            geometry.append(g.to_open3d_geometry())
        return geometry
    
    def sort_by_score(self, reverse = False):
        '''
        输入:
        - reverse: 排序方式, True为从低到高, False为从高到低
        功能:
        - 按照分数对吸取点组排序
        '''
        score = self.suction_group_array[:,0]
        index = np.argsort(score)
        if not reverse:
            index = index[::-1]
        self.suction_group_array = self.suction_group_array[index]
        return self

    def random_sample(self, numSuction = 20):
        '''
        输入:
        - numSuction: 采样的吸取点数量(int)
        输出:
        - 返回采样后的SuctionGroup实例
        '''
        if numSuction > self.__len__():
            raise ValueError('采样数量不能大于吸取点组总数')
        shuffled_suction_group_array = copy.deepcopy(self.suction_group_array)
        np.random.shuffle(shuffled_suction_group_array)
        shuffled_suction_group = SuctionGroup()
        shuffled_suction_group.suction_group_array = copy.deepcopy(shuffled_suction_group_array[:numSuction])
        return shuffled_suction_group

    def nms(self, translation_thresh = 0.1, rotation_thresh = 30.0 / 180.0 * np.pi):
        '''
        输入:
        - translation_thresh: 平移阈值(float)
        - rotation_thresh: 旋转阈值(float, 弧度)
        输出:
        - 返回经过非极大值抑制(NMS)后的SuctionGroup实例
        '''
        from suction_nms import nms_suction
        return SuctionGroup(nms_suction(self.suction_group_array, translation_thresh, rotation_thresh))

