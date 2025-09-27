# 一、PyTorch学习与安装

本文档记录了我学习 PyTorch 框架的完整过程，从环境准备、安装配置到验证使用的全流程。

### 1. PyTorch 简介理解
PyTorch 是一个基于 Python 的科学计算库，支持 GPU 加速的深度学习框架
或者说他是Google官方搭建的一个专门用于机器学习的项目，内含大量目前已经实现的神经网络框架

### 2. 环境准备与硬件检查

要安装pytorch进行模型搭建，需要先检查硬件：GPU是否支持及其版本，然后根据版本选择适配的pytorch版本。
检查GPU版本，需要在命令行中输入以下代码：
```bash
# 查看 NVIDIA 显卡信息
nvidia-smi
```

通过 `nvidia-smi` 查看驱动支持的**最高 CUDA 版本**，然后选择 PyTorch 时需匹配或低于此版本。


# 安装 PyTorch（根据官网生成的命令）
conda install pytorch torchvision torchaudio pytorch-cuda=12.6 -c pytorch -c nvidia
（经过查询GPU版本为12.7，故选择12.6版本）

### 3. 开发环境配置

#### 在PyCharm中的配置

只需在Python解释器中导入torch库即可。

## 学习心得

**环境配置是深度学习的第一步**，正确的安装能避免后续很多问题；**虚拟环境管理**是 Python 开发的重要最佳实践


## 参考资料

[PyTorch 官方文档](https://pytorch.org/docs/)
[PyTorch 安装指南](https://pytorch.org/get-started/locally/)

## 二、ResNet模型学习笔记

### 初始困惑

1. 对深度学习和卷积神经网络基础概念不够了解
2. 不知道如何开始训练一个ResNet模型
3. 时间紧迫，需要快速掌握核心内容

### 寻求的帮助

向DeepSeek助手咨询了：

1.ResNet模型的基本原理
2.训练ResNet所需的工具和环境
3.具体的实施步骤

## 核心知识总结

###ResNet模型关键要点

1. 解决的问题

梯度消失/爆炸：深层网络训练时的数值稳定性问题
网络退化：网络深度增加时性能反而下降的现象

2. 创新点：残差学习

```python
# 残差块的核心公式
输出 = F(x) + x
```

传统网络：直接学习目标映射 H(x)
ResNet：学习残差 F(x) = H(x) - x
优势：如果恒等映射是最优的，直接将残差学习为0即可

3. 网络架构变体

ResNet-18/34：使用基本残差块（两个3×3卷积）
ResNet-50/101/152：使用瓶颈残差块（1×1→3×3→1×1）

**在本次题目中，只需了解并实现ResNet-18模型即可。**

4.技术工具准备

开发环境配置：包括pytorch等必要的库
计划结构：train_process包存储训练过程，classify存储实现分类任务的代码，main函数实现先训练再分类。

数据集选择：CIFAR-10

**代码获取策略**

1. 使用官方示例：PyTorch Examples中的CIFAR分类代码
2. 重点理解：数据加载、模型结构、训练循环三个核心部分
3. 避免从零开始：在有限时间内优先保证流程跑通


## 代码学习记录

### 学习目标

1.掌握PyTorch基础用法和GPU加速训练
2.理解ResNet模型原理和实现
3.完成CIFAR-10图像分类任务
4.学会模型训练、评估和可视化分析


#### PyTorch基础概念理解

#####  张量与梯度计算

```python
# 梯度计算的基本原理
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
y.backward()  # 反向传播计算梯度
print(f"梯度值: {x.grad}")  # 输出: tensor([2.])
```

##### 模型训练的关键步骤

· 前向传播：计算预测结果
· 损失计算：比较预测与真实值
· 反向传播：计算梯度
· 参数更新：优化模型权重

####环境配置挑战

遇到的问题：PyTorch默认安装CPU版本
解决方案：

```bash
# 卸载CPU版本，安装GPU版本
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**遗憾的是不知道为什么，从官网上下载的是CUDA12.6版本，但检测过程中始终显示是CPU版本。这导致只能用CPU进行训练，训练速度过慢，只能训练到一半暂停，这样出来的模型准确率(Accuracy)非常低，基本只有10％多。由于时间紧迫，没有来得及上云计算平台进行训练。**

#### 完整训练流程实现

##### 数据预处理流程

```python
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # 数据增强
    transforms.RandomHorizontalFlip(),         # 数据增强
    transforms.ToTensor(),                     # 转为张量
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
```

##### 训练循环优化

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

##### 模型评估

```python
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():  # 禁用梯度计算，提升性能
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100 * correct / total
```

#### 重要概念深入理解(这是我一开始混淆的)

##### Epoch与Batch关系

```
每个epoch的batch数量 = 训练集样本数 / batch_size
```

CIFAR-10有50,000个训练样本
batch_size=128时，每个epoch有约391个batches

##### 学习率调度策略

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# 每30个epoch学习率乘以0.1
```

##### 模型保存与加载

```python
# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 加载模型
model.load_state_dict(torch.load('model.pth'))
```

### 关键收获
1. GPU加速重要性：正确配置GPU环境可大幅提升训练效率
2. 数据预处理关键性：合适的数据增强和标准化对模型性能影响显著
3. 梯度管理：训练时需要梯度，评估时应使用torch.no_grad()优化性能

### 训练结果

运行main函数，可以查看模型的训练参数，如每一段Epoch的Loss和Accuracy等，分类后可以查看这两个参数并打印出可视化图片(存储在与main函数同目录下)。

**通过这次系统学习，我不仅掌握了ResNet模型的实际应用，还建立了完整的深度学习项目开发认知。**
