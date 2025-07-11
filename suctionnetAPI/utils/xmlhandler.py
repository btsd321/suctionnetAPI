__author__ = 'Minghao Gou'
__version__ = '1.0'

from xml.etree.ElementTree import Element, SubElement, tostring
import xml.etree.ElementTree as ET
import xml.dom.minidom
from transforms3d.quaternions import mat2quat
from transforms3d.euler import quat2euler
import numpy as np
from .trans3d import get_mat
import os
from .pose import pose_list_from_pose_vector_list


class xmlWriter():
    def __init__(self, topfromreader=None):
        # 初始化xmlWriter对象, 可选传入已有的XML根节点
        self.topfromreader = topfromreader
        self.poselist = []
        self.objnamelist = []
        self.objpathlist = []
        self.objidlist = []
    def addobject(self, pose, objname, objpath, objid):
        # 添加一个物体的信息, 包括位姿、名称、路径和id
        # pose为4x4位姿矩阵
        self.poselist.append(pose)
        self.objnamelist.append(objname)
        self.objpathlist.append(objpath)
        self.objidlist.append(objid)

    def objectlistfromposevectorlist(self, posevectorlist, objdir, objnamelist, objidlist):
        # 根据位姿向量列表和物体信息, 批量添加物体
        self.poselist = []
        self.objnamelist = []
        self.objidlist = []
        self.objpathlist = []
        for i in range(len(posevectorlist)):
            id, x, y, z, alpha, beta, gamma = posevectorlist[i]
            objname = objnamelist[objidlist[i]]
            self.addobject(get_mat(x, y, z, alpha, beta, gamma),
                           objname, os.path.join(objdir, objname), id)

    def writexml(self, xmlfilename='scene.xml'):
        # 将当前所有物体信息写入XML文件
        if self.topfromreader is not None:
            self.top = self.topfromreader
        else:
            self.top = Element('scene')
        for i in range(len(self.poselist)):
            obj_entry = SubElement(self.top, 'obj')

            obj_name = SubElement(obj_entry, 'obj_id')
            obj_name.text = str(self.objidlist[i])

            obj_name = SubElement(obj_entry, 'obj_name')
            obj_name.text = self.objnamelist[i]

            obj_path = SubElement(obj_entry, 'obj_path')
            obj_path.text = self.objpathlist[i]
            pose = self.poselist[i]
            pose_in_world = SubElement(obj_entry, 'pos_in_world')
            pose_in_world.text = '{:.4f} {:.4f} {:.4f}'.format(
                pose[0, 3], pose[1, 3], pose[2, 3])

            rotationMatrix = pose[0:3, 0:3]
            quat = mat2quat(rotationMatrix)

            ori_in_world = SubElement(obj_entry, 'ori_in_world')
            ori_in_world.text = '{:.4f} {:.4f} {:.4f} {:.4f}'.format(
                quat[0], quat[1], quat[2], quat[3])
        xmlstr = xml.dom.minidom.parseString(
            tostring(self.top)).toprettyxml(indent='    ')
        # 移除空白行
        xmlstr = "".join([s for s in xmlstr.splitlines(True) if s.strip()])
        with open(xmlfilename, 'w') as f:
            f.write(xmlstr)
            #print('log:write annotation file '+xmlfilename)


class xmlReader():
    def __init__(self, xmlfilename):
        # 初始化xmlReader对象, 读取指定XML文件
        self.xmlfilename = xmlfilename
        etree = ET.parse(self.xmlfilename)
        self.top = etree.getroot()

    def showinfo(self):
        # 打印当前XML文件中所有物体的名称信息
        print('Resumed object(s) already stored in '+self.xmlfilename+':')
        for i in range(len(self.top)):
            print(self.top[i][1].text)

    def gettop(self):
        # 返回XML根节点
        return self.top

    def getposevectorlist(self):
        # 解析XML, 返回所有物体的位姿向量列表
        # 位姿向量格式: [objectid, x, y, z, alpha, beta, gamma]
        posevectorlist = []
        for i in range(len(self.top)):
            objectid = int(self.top[i][0].text)
            objectname = self.top[i][1].text
            objectpath = self.top[i][2].text
            translationtext = self.top[i][3].text.split()
            translation = []
            for text in translationtext:
                translation.append(float(text))
            quattext = self.top[i][4].text.split()
            quat = []
            for text in quattext:
                quat.append(float(text))
            alpha, beta, gamma = quat2euler(quat)
            x, y, z = translation
            alpha *= (180.0 / np.pi)
            beta *= (180.0 / np.pi)
            gamma *= (180.0 / np.pi)
            posevectorlist.append([objectid, x, y, z, alpha, beta, gamma])
        return posevectorlist

    def get_pose_list(self):
        # 返回所有物体的Pose对象列表
        pose_vector_list = self.getposevectorlist()
        return pose_list_from_pose_vector_list(pose_vector_list)

def empty_pose_vector(objectid):
    # 生成一个指定物体id的默认位姿向量
    # [object id, x, y, z, alpha, beta, gamma]
    # alpha, beta, gamma为欧拉角(角度制)
    return [objectid, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0]

def empty_pose_vector_list(objectidlist):
    # 批量生成默认位姿向量列表
    pose_vector_list = []
    for id in objectidlist:
        pose_vector_list.append(empty_pose_vector(id))
    return pose_vector_list

def getposevectorlist(objectidlist, is_resume, num_frame, frame_number, xml_dir):
    # 根据是否恢复和文件存在性, 获取指定帧的物体位姿向量列表
    if not is_resume or (not os.path.exists(os.path.join(xml_dir, '%04d.xml' % num_frame))):
        print('log:create empty pose vector list')
        return empty_pose_vector_list(objectidlist)
    else:
        print('log:resume pose vector from ' +
              os.path.join(xml_dir, '%04d.xml' % num_frame))
        xmlfile = os.path.join(xml_dir, '%04d.xml' % num_frame)
        mainxmlReader = xmlReader(xmlfile)
        xmlposevectorlist = mainxmlReader.getposevectorlist()
        posevectorlist = []
        for objectid in objectidlist:
            posevector = [objectid, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for xmlposevector in xmlposevectorlist:
                if xmlposevector[0] == objectid:
                    posevector = xmlposevector
            posevectorlist.append(posevector)
        return posevectorlist

def getframeposevectorlist(objectidlist, is_resume, frame_number, xml_dir):
    # 获取所有帧的物体位姿向量列表(每帧一个列表)
    frameposevectorlist = []
    for num_frame in range(frame_number):
        if not is_resume or (not os.path.exists(os.path.join(xml_dir,'%04d.xml' % num_frame))):
            posevectorlist=getposevectorlist(objectidlist,False,num_frame,frame_number,xml_dir)	
        else:
            posevectorlist=getposevectorlist(objectidlist,True,num_frame,frame_number,xml_dir)
        frameposevectorlist.append(posevectorlist)
    return frameposevectorlist
