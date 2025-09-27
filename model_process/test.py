#该部分用于评估训练模型的效果

import torch

def test(model, test_loader, criterion, device):
    model.eval()  #设置为评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  #禁用梯度计算
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            test_value, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total

    #返回评估值Loss和Accuracy
    return test_loss, test_acc