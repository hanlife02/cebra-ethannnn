# CEBRA模型比较分析

## 模型概述

您展示的代码使用CEBRA（Consistent Embedding of high-dimensional Recordings using Auxiliary variables）框架训练了三种不同的神经数据嵌入模型：

1. **cebra_pos_model** - 使用手部位置标签训练
2. **cebra_target_model** - 使用目标方向标签训练
3. **cebra_time_model** - 使用时间信息训练（无行为标签）

## 三个模型的主要区别

### 1. 训练数据和标签类型

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

### 2. 条件设置差异

- **cebra_pos_model** 和 **cebra_target_model**:
  - `conditional='time_delta'` - 使用时间差作为条件约束
  - 这意味着模型学习时会考虑时间上相近的样本应该在嵌入空间中也相近

- **cebra_time_model**:
  - `conditional='time'` - 直接使用时间作为条件约束
  - 模型完全依赖时间结构来学习神经活动的表示

### 3. 时间偏移设置

- **cebra_pos_model** 和 **cebra_target_model**:
  - `time_offsets=10` - 使用较大的时间窗口

- **cebra_time_model**:
  - `time_offsets=5` - 使用较小的时间窗口
  - 这可能使模型对短时间内的神经活动变化更敏感

## 可视化结果分析

从您提供的可视化代码可以看出：

1. **cebra_pos_model** 的嵌入结果与手部位置（x和y坐标）高度相关，可以在3D嵌入空间中清晰地看到位置信息的编码。

2. **cebra_target_model** 的嵌入结果能够区分不同的目标方向，并且通过方向平均的嵌入可以看到不同方向在嵌入空间中的分布。

3. **cebra_time_model** 虽然没有使用任何行为标签训练，但其嵌入结果仍然能够在一定程度上反映手部位置信息，这表明时间结构本身包含了与行为相关的信息。

## 模型应用场景

- **cebra_pos_model**: 适合研究神经活动与连续运动变量（如位置）之间的关系。
- **cebra_target_model**: 适合研究神经活动与离散行为类别（如运动方向）之间的关系。
- **cebra_time_model**: 适合在没有行为标签的情况下探索神经活动的内在动态结构。

## 总结

这三个模型展示了CEBRA框架的灵活性，能够根据不同类型的辅助信息（连续标签、离散标签或仅时间）学习神经数据的低维表示。最后一个cell中的`compare_models`函数会提供这三个模型性能的定量比较，帮助评估哪种类型的辅助信息对于理解神经活动模式最有效。