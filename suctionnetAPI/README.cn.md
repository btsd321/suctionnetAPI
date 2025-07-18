# SuctionNet-1Billion 数据集 API

本项目为 RA-L 论文《SuctionNet-1Billion:  A  Large-Scale  Benchmark  for  Suction  Grasping》中 SuctionNet-1Billion 数据集的官方 API。

## 数据集

请从我们的 [SuctionNet 官方网页](https://graspnet.net/suction) 下载数据和标签文件。

## 吸取点定义

一个吸取点由其三维坐标和吸取方向共同定义。吸取方向是一个归一化的向量，指向物体表面外侧。如下图所示。

在评估你的算法时，需将每个预测的吸取点表示为一个 7 维向量。第一个元素为预测得分，接下来的三个元素为三维吸取点坐标，最后三个元素为归一化的吸取方向。对于每个视角(每个场景共 256 个视角)，假设你预测了 N 个吸取点，则该视角的结果应保存为一个 `Nx7` 的 numpy 数组。

<img src="https://github.com/graspnet/suctionnetAPI/blob/master/suction_definition.jpg" />

## 安装方法

请先安装 [Point Cloud Utils](https://github.com/fwilliams/point-cloud-utils)，然后执行以下命令安装本 API：

``` 
git clone https://github.com/graspnet/suctionnetAPI
cd suctionnetAPI
pip install .
```

## 评测前置条件

在进行预测结果评测前，请确保通过了数据完整性检查。可参考 `examples/check_and_explore_data.py` 脚本进行检查。

## 示例

我们在 `examples` 文件夹中提供了多个 API 使用示例：

- 检查、探索和加载数据：`examples/check_and_explore_data.py`
- 评测你的结果：`examples/evaluation.py`
- 可视化数据和标签：`visualization.py`
- 创建稠密点云：`dense_pcd.py`

## 论文引用

如果本项目对你的研究有帮助，请引用如下论文：

```
@ARTICLE{suctionnet,
  author={Cao, Hanwen and Fang, Hao-Shu and Liu, Wenhai and Lu, Cewu},
  journal={IEEE Robotics and Automation Letters}, 
  title={SuctionNet-1Billion: A Large-Scale Benchmark for Suction Grasping}, 
  year={2021},
  volume={6},
  number={4},
  pages={8718-8725},
  doi={10.1109/LRA.2021.3115406}}
```