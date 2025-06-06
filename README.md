# 2024-2025春季《神经网络的计算基础》期末作业

> [!IMPORTANT]
> - 本repo基于论文[《Learnable latent embeddings for joint behavioural and neural analysis》](https://www.nature.com/articles/s41586-023-06031-6) ，为了复现其中的关键实验和图表而创建。

## 项目结构

```

├── Decoding_movie_features_optimize.ipynb // 改进优化器后的解码小鼠的文件
│
├── Figures-re  // 用于复现论文中五处图片中大部分内容
│   ├── figure1.ipynb
│   ├── figure2.ipynb
│   ├── figure3.ipynb
│   ├── figure4.ipynb
│   └── figure5.ipynb
│
├── README.md
│
├── data  //实验需要的，已经预处理好的数据
│   ├── Figure1.h5
│   ├── Figure2.h5
│   ├── Figure3.h5
│   ├── Figure4Revision.h5
│   ├── Figure5Revision.h5
│   ├── SupplVideo1.h5
│   ├── allen
│   │   ├── allen_movie1_neuropixel
│   │   ├── data_summary.csv
│   │   ├── features
│   │   ├── figures
│   │   └── visual_drift
│   ├── autolfads_decoding_2d_full.csv
│   ├── figure2_pivae_mcmc.csv
│   ├── monkey_reaching_preload_smth_40
│   ├── rat_hippocampus
│   ├── results_v1
│   ├── results_v3
│   └── synthetic
│
├── dataset.md // 关于实验数据集的名称汇总，便于查找
│
├── figures // 复现论文过程中保存了一些重要图像
│
├── monkey_reaching_all.ipynb // 用CEBRA模型训练猴子前肢和行为的数据集
│
├── result
│   └── monkey_reaching_all_embeddings.npz // 训练后保存已训练好的猴子前肢和行为数据集
│
├──environment.yml // 所有环境配置
│
├── test // 实现最基础实验功能的代码，未经改进
    ├── Decoding_movie_features.ipynb
    └── monkey_reaching_active.ipynb

```

## 关于环境

原repo里的依赖存在各种版本错误，和某些依赖缺失的问题。

特此将能实现本repo所需要的依赖打包到`environment.yml`文件里

你可以运行下面的命令安装运行本repo的所有依赖

```shell
conda env create -f environment.yml
```

## 两个主要的代码文件


### 1. monkey_reaching_all.ipynb

使用CEBRA框架训练了三种不同的神经数据嵌入模型：

1. **cebra_pos_model** - 使用手部位置标签训练
2. **cebra_target_model** - 使用目标方向标签训练
3. **cebra_time_model** - 使用时间信息训练（无行为标签）

#### 三个模型的主要区别

##### 1. 训练数据和标签类型

- **cebra_pos_model**:
  - 使用`pos_dataset.neural`作为输入数据
  - 使用`pos_dataset.continuous_index`作为连续标签（手部位置坐标）
  - 标签是连续值，表示x和y坐标

- **cebra_target_model**:
  - 使用`target_dataset.neural`作为输入数据
  - 使用`target_dataset.discrete_index`作为离散标签（目标方向）
  - 标签是离散值，表示8个不同的方向类别

- **cebra_time_model**:
  - 使用`target_dataset.neural`作为输入数据
  - 不使用任何外部标签，仅依赖时间结构
  - 使用时间作为隐式标签

##### 2. 条件设置差异

- **cebra_pos_model** 和 **cebra_target_model**:
  - `conditional='time_delta'` - 使用时间差作为条件约束
  - 这意味着模型学习时会考虑时间上相近的样本应该在嵌入空间中也相近

- **cebra_time_model**:
  - `conditional='time'` - 直接使用时间作为条件约束
  - 模型完全依赖时间结构来学习神经活动的表示

##### 3. 时间偏移设置

- **cebra_pos_model** 和 **cebra_target_model**:
  - `time_offsets=10` - 使用较大的时间窗口

- **cebra_time_model**:
  - `time_offsets=5` - 使用较小的时间窗口
  - 这可能使模型对短时间内的神经活动变化更敏感

##### 可视化结果分析

从您提供的可视化代码可以看出：

1. **cebra_pos_model** 的嵌入结果与手部位置（x和y坐标）高度相关，可以在3D嵌入空间中清晰地看到位置信息的编码。

2. **cebra_target_model** 的嵌入结果能够区分不同的目标方向，并且通过方向平均的嵌入可以看到不同方向在嵌入空间中的分布。

3. **cebra_time_model** 虽然没有使用任何行为标签训练，但其嵌入结果仍然能够在一定程度上反映手部位置信息，这表明时间结构本身包含了与行为相关的信息。

##### 模型应用场景

- **cebra_pos_model**: 适合研究神经活动与连续运动变量（如位置）之间的关系。
- **cebra_target_model**: 适合研究神经活动与离散行为类别（如运动方向）之间的关系。
- **cebra_time_model**: 适合在没有行为标签的情况下探索神经活动的内在动态结构。

##### 对比模型和数据集

- **compute_model_consistency** 对比了不同数据集和模型下的一致性

#### 总结

这三个模型展示了CEBRA框架的灵活性，能够根据不同类型的辅助信息（连续标签、离散标签或仅时间）学习神经数据的低维表示。

### 2. Decoding_movie_features_optimize.ipynb

展示了如何使用CEBRA对Allen Institute的视觉皮层神经活动数据进行降维和解码分析。

#### 主要功能模块

1. 数据加载与准备
使用Allen Institute的电影刺激数据集
包含两种记录模式：Ca²⁺成像数据和Neuropixels电生理数据
支持多个视觉皮层区域：VISp, VISpm, VISam, VISrl, VISal, VISl
2. 数据可视化
神经活动热图展示（Ca²⁺和Neuropixels数据）
t-SNE降维可视化
CEBRA嵌入结果的散点图
3. CEBRA模型训练
包含三种训练方式：
单会话训练：分别训练Ca²⁺和Neuropixels数据
联合训练：同时训练两种模态的数据
跨皮层区域训练：比较不同皮层区域的表征
4. 优化器配置
使用AdamW优化器替代原始的Adam
添加学习率调度器(OneCycleLR)
实现梯度裁剪
配置权重衰减等正则化技术
5. 解码分析
帧ID解码：使用KNN和贝叶斯分类器预测视频帧
基线比较：原始神经数据 vs CEBRA嵌入
性能评估：计算解码准确率
6. 一致性分析
皮层内一致性：同一皮层区域不同模态间的线性关系
皮层间一致性：不同皮层区域间的表征相似性
技术特点
多模态融合：整合Ca²⁺成像和电生理数据
优化训练策略：改进的优化器配置提升训练稳定性
跨区域分析：探索不同视觉皮层区域的功能表征
定量评估：通过解码任务验证嵌入质量

#### 总结

这个notebook为神经科学研究提供了一个完整的分析流程，可以用于：

理解视觉皮层的信息处理机制
比较不同记录技术的表征能力
评估多模态数据融合的效果
优化神经数据的降维和表征学习方法

## 关于复现与创新

关于`Figures-re`目录下的五个文件，成功复现论文的五个图形

关于`Decoding_movie_features_optimize.ipynb`，改进了优化器部分

关于`monkey_reaching_all.ipynb`，根据API文档独立编