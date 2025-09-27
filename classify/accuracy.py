# 分析每个类别的分类性能
def analyze(predictions, labels):
    #定义CIFAR-10类别名称
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    #创建记录表
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    #将每个类别的准确率记录
    for i in range(len(predictions)):
        label = labels[i]
        class_correct[label] += (predictions[i] == label)
        class_total[label] += 1

    #打印每个类别的准确率
    print("\n每个类别的分类准确率:")
    for i in range(10):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'{classes[i]:>12s}: {accuracy:.2f}%')
        else:
            print(f'{classes[i]:>12s}: 无样本')