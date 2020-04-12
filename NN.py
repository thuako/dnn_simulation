import torch
import torch.nn as nn

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import visdom

vis = visdom.Visdom()
vis.close(env="main")

def value_tracker(value_plot, value, num):
    '''num, loss_value, are Tensor'''
    vis.line(X=num,Y=value,win = value_plot,update='append')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device =='cuda':
    torch.cuda.manual_seed_all(777)


'''data load'''
trainset = torchvision.datasets.MNIST(root='MNIST_Data/', train=True,download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='MNIST_Data/', train=True,download=False, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=256,shuffle=False, num_workers=0)

dataiter = iter(trainloader)
images, labels = dataiter.next()
vis.images(images)


class NN(nn.Module):

    def __init__(self,feature, Dropout = 0.3,num_classes=1000):
        super(NN, self).__init__()
        self.feature = feature
        self.Dropout = Dropout
        self.num_classes = num_classes
        self.layer = self.make_layer(feature, Dropout, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer(x)
        return x

    def make_layer(self, feature, D, n):
        layers = []
        in_channel = 784
        for v in feature:
            layers += [nn.Linear(in_channel, v*v, bias=True)]
            layers += [nn.BatchNorm1d(v*v)]
            layers += [nn.ReLU()]
            layers += [nn.Dropout(p=D)]
            in_channel = v*v
        layers += [nn.Linear(feature[-1]*feature[-1], n)]
        return nn.Sequential(*layers)

model1 = NN([16, 16, 8, 8], 0.3, 10).to(device)
model2 = NN([16, 16, 16, 8], 0.3, 10).to(device)
model3 = NN([16, 16, 16, 16], 0.3, 10).to(device)


criterion1 = nn.CrossEntropyLoss().to(device)
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.1)

criterion2 = nn.CrossEntropyLoss().to(device)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)

criterion3 = nn.CrossEntropyLoss().to(device)
optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.1)

epochs=5

loss_plt1 = []
loss_plt2 = []
loss_plt3 = []

acc_plt1 = []
acc_plt2 = []
acc_plt3 = []

def acc_check(net, test_set, dataset):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_set:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = (100 * correct / total)
    print('Accuracy of the network on the 10000 test images: %d %%' % acc)
    dataset.append([acc])
    return acc

for epoch in range(epochs):
    running_loss1, running_loss2, running_loss3 = 0, 0, 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        outputs1 = model1(inputs)
        outputs2 = model2(inputs)
        outputs3 = model3(inputs)

        loss1 = criterion1(outputs1, labels)
        loss2 = criterion2(outputs1, labels)
        loss3 = criterion3(outputs1, labels)

        loss1.backward()
        loss2.backward()
        loss3.backward()

        optimizer1.step()
        optimizer2.step()
        optimizer3.step()

        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
        running_loss3 += loss3.item()

        if i%30 == 29:
            loss_plt1.append([running_loss1 / 30])
            loss_plt2.append([running_loss2 / 30])
            loss_plt3.append([running_loss3 / 30])

            print('[%d, %5d] vgg loss: %.3f' % (epoch + 1, i + 1, running_loss1 / 30))
            print('[%d, %5d] vgg loss: %.3f' % (epoch + 1, i + 1, running_loss2 / 30))
            print('[%d, %5d] vgg loss: %.3f' % (epoch + 1, i + 1, running_loss3 / 30))

    acc1 = acc_check(model1, testloader, acc_plt1)
    acc2 = acc_check(model2, testloader, acc_plt2)
    acc3 = acc_check(model3, testloader, acc_plt3)

def plot_compare(loss_list1:list, loss_list2:list, loss_list3:list, ylim=None, title=None,)->None:
    plt.figure(figsize=(15, 20))
    plt.plot(loss_list1, label = 'model1')
    plt.plot(loss_list2, label='model2')
    plt.plot(loss_list3, label='model3')

    if ylim:
        plt.ylim(ylim)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid('on')
    plt.show()
    
plot_compare(loss_plt1, loss_plt2, loss_plt3, [0, 1.0])
plot_compare(acc_plt1, acc_plt2, acc_plt3,)
