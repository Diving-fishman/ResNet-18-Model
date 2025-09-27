#用于实现训练结果可视化

import matplotlib.pyplot as plt
import numpy as np
import torch

#可视化一些分类结果
def visualize(model, testset, num_images=12):
    #设置设备
    device = next(model.parameters()).device

    #随机选择一些测试图像
    indices = np.random.choice(len(testset), num_images, replace=False)

    #创建图像网格
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()

    #创建CIFAR-10类别名称
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    model.eval()
    #禁止梯度
    with torch.no_grad():
        for i, idx in enumerate(indices):
            #获取图像和真实标签
            image, true_label = testset[idx]

            #进行预测
            input_tensor = image.unsqueeze(0).to(device)
            output = model(input_tensor)
            output_receive , predicted = torch.max(output.data, 1)    #仅关心预测结果
            predicted_label = predicted.item()

            #反标准化图像以正常显示
            image = image.numpy().transpose((1, 2, 0))
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            image = std * image + mean
            image = np.clip(image, 0, 1)

            #显示图像
            axes[i].imshow(image)

            #设置标题颜色：绿色表示正确，红色表示错误
            color = 'green' if predicted_label == true_label else 'red'
            axes[i].set_title(f'True: {classes[true_label]}\nPred: {classes[predicted_label]}',
                              color=color, fontsize=10)
            axes[i].axis('off')

    #可视化结果
    plt.tight_layout()
    plt.savefig('classification_results.png', dpi=300, bbox_inches='tight')
    plt.show()