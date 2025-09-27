#整个ResNet-18模型训练的主函数，包括了训练和分类任务两部分

import torch
from model_process.model import model_c
from model_process.train import train
from model_process.test import test
from model_process.data import train_loader, test_loader
from classify.main_class import main_class

#主函数
def main():

    #设置训练参数，结果参数置零

    num_epochs = 10
    best_acc = 0

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    #创建模型
    model, criterion, optimizer, scheduler, device = model_c()

    #训练循环
    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch: {epoch}')

        #训练
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch, device)

        #测试
        test_loss, test_acc = test(model, test_loader, criterion, device)

        #更新学习率
        scheduler.step()

        #记录结果
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

        #保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'resnet18_process.pth')

    #保存最终模型
    save_dir = "classify"
    model_path = f"{save_dir}/resnet18.pth"
    torch.save(model.state_dict(), model_path)
    print(f'Best accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()
    main_class()