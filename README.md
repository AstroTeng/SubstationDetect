# 变电站拓扑图元检测与识别系统

本项目是一个基于OpenCV和YOLO的变电站拓扑图元检测与识别系统，包含图元生成、神经网络训练和拓扑结构分析等功能模块。

## 功能特性

- **图元生成引擎**
  - 自动生成6类电力图元（变压器/隔离开关等）
  - 支持随机旋转、噪声添加、颜色变换
  - 批量生成训练数据集

- **深度学习检测**
  - 自定义CNN神经网络架构
  - YOLOv4-tiny目标检测集成
  - 支持模型训练/保存/加载

- **拓扑结构分析**
  - 图元间连接关系检测
  - 母线自动识别
  - 邻接矩阵可视化

## 环境依赖

- OpenCV 4.x
- Darknet (YOLO实现)
- CUDA 10.2+ (可选)
- CUDNN 8.x (可选)
- C++17兼容编译器

## 安装说明

1. 克隆仓库
```bash
git clone https://github.com/yourrepo/substation-topo-detection.git
cd substation-topo-detection
```

2. 安装依赖
```bash
# OpenCV
sudo apt install libopencv-dev

# Darknet
git clone https://github.com/AlexeyAB/darknet.git
cd darknet && make
```

3. 编译项目
```bash
mkdir build && cd build
cmake ..
make
```

## 使用指南

### 1. 图元生成
```bash
./TopoDetect --generate 1000  # 生成1000个样本
```
样本保存在`pics/generated-elements/`

### 2. 模型训练
```bash
./TopoDetect --train --epoch 50 --batch 32
```
权重文件保存在`data-new/`

### 3. 拓扑检测
```bash
./TopoDetect --detect test.jpg
```
支持图片和视频输入，检测结果实时显示

### 4. 交互模式
```bash
./TopoDetect
```
通过控制台菜单选择操作：
- 0) 随机初始化权重
- 1) 加载权重文件
- 2) 保存当前权重
- 3) 执行单图预测
- 5) 启动训练流程
- 6) 执行测试集验证

## 项目结构

```
.
├── data/                # YOLO配置文件
│   ├── obj.cfg
│   └── yolov4-tiny-obj_last.weights
├── pics/                # 图像数据集
│   ├── generated-elements/  # 生成图元
│   └── test-new/            # 测试结果
├── src/                 # 源代码
│   ├── CoutShape.cpp    # 绘图模块
│   ├── ElementGenerate.cpp # 图元生成
│   ├── NeuronNet.cpp    # 神经网络
│   └── TopoDetect.cpp   # 主程序
└── CMakeLists.txt
```

## 算法流程

1. **图像预处理**
   - Canny边缘检测
   - 形态学操作（膨胀/腐蚀）
   - 连通域分析

2. **目标检测**
   - YOLO识别电力图元
   - 质心修正定位

3. **拓扑分析**
   - 连接关系建模
   - 邻接矩阵构建
   - 母线自动识别

## 常见问题

Q: 检测时出现CUDA内存错误  
A: 尝试减小输入分辨率或使用CPU模式编译Darknet

Q: 图元生成效果不理想  
A: 调整`ElementGenerate.cpp`中的噪声参数和变换范围

Q: 模型训练不收敛  
A: 检查学习率设置，增加训练样本多样性

## 许可协议

本项目采用MIT许可证，详情见LICENSE文件。
