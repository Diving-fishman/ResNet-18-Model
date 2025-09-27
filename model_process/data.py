#该部分用于应用Cifar-10的数据集用来训练模型

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

#数据预处理：裁剪、标准化
transform = transforms.Compose([
    transforms.Resize(256),           # 调整尺寸
    transforms.CenterCrop(224),       # 中心裁剪到224x224
    transforms.ToTensor(),            # 转为Tensor
    transforms.Normalize(             # 标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#应用Cifar-10中的数据集
train_dataset = datasets.CIFAR10(
    root='./data',       #下载到当前目录的data文件夹内
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

#创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)