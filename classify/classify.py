#进行分类任务
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

#定义分类函数
def classify(model_path, num_classes=10):
    #设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'resnet18.pth'  #替换为已经训练好的模型

    #加载模型结构
    model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)

    #添加resize变换将32x32图像放大到224x224（标准ResNet的输入尺寸）
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    #加载训练好的权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    #设置为评估模式
    model.eval()

    #加载CIFAR-10测试集
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    #创建CIFAR-10类别名称
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    #在整个测试集上评估模型
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #保存预测结果用于进一步分析
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    #计算整体准确率
    accuracy = 100 * correct / total
    print(f'模型在CIFAR-10测试集上的准确率: {accuracy:.2f}%')

    #返回评估参数
    return model, testset, all_predictions, all_labels, accuracy

#利用定义的函数进行分类

#使用训练好的模型进行分类
model_path = 'resnet18.pth'
model, testset, predictions, labels, accuracy = classify(model_path)