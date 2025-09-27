#分类任务的主函数部分

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from .visualize import visualize
from .accuracy import analyze


def main_class():
    #加载训练好的模型
    model_path = 'resnet18.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #创建模型结构
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)

    #加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    #准备数据，设置参数和训练时相同
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    #在整个测试集上评估
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    #禁止梯度
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'模型准确率: {accuracy:.2f}%')

    #调用visualize函数进行可视化

    #可视化分类结果
    visualize(model, testset)

    #分析每个类别的性能
    analyze(all_predictions, all_labels)