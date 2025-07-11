# SuctionNet-1Billion 数据集API

本API用于RA-L论文《SuctionNet-1Billion:  A  Large-Scale  Benchmark  for  Suction  Grasping》中SuctionNet-1Billion数据集的访问与处理。

## 数据集

请从我们的[SuctionNet官网](https://graspnet.net/suction)下载数据和标签文件。

## 吸取点定义

一个吸取点由其三维空间中的吸取位置和方向定义。方向为归一化的法向量，指向物体表面外侧。如下图所示。

在评估你的算法时，需将每个预测的吸取点表示为7维向量。第1维为预测得分，接下来的3维为吸取点的三维坐标，最后3维为归一化的吸取方向。对于每个视角(每个场景共256个视角)，假设你预测了`N`个吸取点，则该视角的结果应保存为`Nx7`的numpy数组。

<img src="https://github.com/graspnet/suctionnetAPI/blob/master/suction_definition.jpg" />

## 安装方法

请先安装 [Point Cloud Utils](https://github.com/fwilliams/point-cloud-utils)，然后执行以下命令安装本API：

``` 
git clone https://github.com/graspnet/suctionnetAPI
cd suctionnetAPI
pip install .
```

## 评测前准备

在进行预测评测前，请确保数据完整性通过检查。可参考 `examples/check_and_explore_data.py` 脚本进行完整性检查。

## 示例说明

我们在`examples`文件夹中提供了多个API使用示例：

- 检查、浏览和加载数据：`examples/check_and_explore_data.py`
- 评估预测结果：`examples/evaluation.py`
- 可视化数据和标签：`visualization.py`
- 生成稠密点云：`dense_pcd.py`

## 参考文献

如果本工作对您的研究有帮助，请引用如下论文：

```
@ARTICLE{suctionnet,
  author={Cao, Hanwen and Fang, Hao-Shu and Liu, Wenhai and Lu, Cewu},
  journal={IEEE Robotics and Automation Letters}, 
  title={SuctionNet-1Billion: A Large-Scale Benchmark for Suction Grasping}, 
  year={2021},
  volume={6},
  number={4},
  pages={8718-8725},
  doi=