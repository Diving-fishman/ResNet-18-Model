#该部分用于训练模型

import torch

def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()  #设置为训练模式
    #设置训练参数为0
    running_loss = 0.0
    correct = 0
    total = 0

    #训练循环
    for batch_num, (inputs, targets) in enumerate(train_loader):
        #确保数据集上传到GPU进行训练
        inputs, targets = inputs.to(device), targets.to(device)

        #检查CUDA是否可用
        print(f"CUDA可用: {torch.cuda.is_available()}")

        #检查GPU数量
        print(f"GPU数量: {torch.cuda.device_count()}")

        #检查当前GPU
        print(f"当前GPU: {torch.cuda.current_device()}")

        #检查GPU名称
        if torch.cuda.is_available():
            print(f"GPU名称: {torch.cuda.get_device_name(0)}")

        #前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        #反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #统计信息
        running_loss += loss.item()
        train_value, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        #每100个batch打印一次，便于查看进度
        if batch_num % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_num}, Loss: {loss.item():.3f}')

    #计算本周期的平均损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    #返回每一周期的参数
    return epoch_loss, epoch_acc