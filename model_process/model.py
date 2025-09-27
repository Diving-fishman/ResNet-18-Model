#该部分用于创建模型

import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn

def model_c():
    # 创建ResNet-18模型，适配Cifar-10的10个类别
    model = models.resnet18(pretrained=False, num_classes=10)

    #使用GPU训练
    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #创建损失函数
    criterion = nn.CrossEntropyLoss()

    #创建优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    #创建学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    #函数返回训练模型和上述的创建的参数
    return model, criterion, optimizer, scheduler , device