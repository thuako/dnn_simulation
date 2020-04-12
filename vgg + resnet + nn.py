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

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)

train_data_mean = trainset.data.mean( axis=(0,1,2) )
train_data_std = trainset.data.std( axis=(0,1,2) )
train_data_mean = train_data_mean / 255
train_data_std = train_data_std / 255
'''data load'''
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.ToTensor(),transforms.Normalize(train_data_mean, train_data_std)])
transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(train_data_mean, train_data_std)])

trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')



###########################################################################

#resnet make layer
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x

        out = self.conv1(x) # 3x3 stride = 2
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # 3x3 stride = 1
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes) #conv1x1(64,64)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)#conv3x3(64,64)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion) #conv1x1(64,256)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x) # 1x1 stride = 1
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # 3x3 stride = stride
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) # 1x1 stride = 1
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x.shape =[1, 16, 32,32]
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        # x.shape =[1, 128, 32,32]
        x = self.layer2(x)
        # x.shape =[1, 256, 32,32]
        x = self.layer3(x)
        # x.shape =[1, 512, 16,16]
        x = self.layer4(x)
        # x.shape =[1, 1024, 8,8]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


##############################
#vgg make layer
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)
cfg = [32,32,'M', 64,64,128,128,128,'M',256,256,256,512,512,512,'M'] #13 + 3 =vgg16

###################################

resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], 10, True).to(device)
vgg16= VGG(make_layers(cfg),10,True).to(device)


res_criterion = nn.CrossEntropyLoss().to(device)
res_optimizer = torch.optim.SGD(resnet50.parameters(), lr = 0.1, momentum = 0.9, weight_decay=5e-4)
res_lr_sche = optim.lr_scheduler.StepLR(res_optimizer, step_size=10, gamma=0.5)
vgg_criterion = nn.CrossEntropyLoss().to(device)
vgg_optimizer = torch.optim.SGD(vgg16.parameters(), lr = 0.005,momentum=0.9)
vgg_lr_sche = optim.lr_scheduler.StepLR(vgg_optimizer, step_size=5, gamma=0.9)


#####################################################################################

vgg_loss_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='vgg_loss_tracker', legend=['loss'], showlegend=True))
res_loss_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='res_loss_tracker', legend=['loss'], showlegend=True))

vgg_acc_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='vgg_acc_tracker', legend=['acc'], showlegend=True))
res_acc_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='res_acc_tracker', legend=['acc'], showlegend=True))


vgg_loss_matplt = []
res_loss_matplt = []
vgg_acc_matplt = []
res_acc_matplt = []

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
    dataset.append(acc)
    return acc

epochs = 5
for epoch in range(epochs):  # loop over the dataset multiple times
    vgg_running_loss, res_running_loss = 0.0, 0.0
    res_lr_sche.step()
    vgg_lr_sche.step()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        vgg_optimizer.zero_grad()
        res_optimizer.zero_grad()

        # forward + backward + optimize
        res_outputs = resnet50(inputs)
        vgg_outputs = vgg16(inputs)

        res_loss = res_criterion(res_outputs, labels)
        vgg_loss = vgg_criterion(vgg_outputs, labels)

        res_loss.backward()
        vgg_loss.backward()

        res_optimizer.step()
        vgg_optimizer.step()

        # print statistics
        res_running_loss += res_loss.item()
        vgg_running_loss += vgg_loss.item()

        if i % 30 == 29:  # print every 30 mini-batches
            value_tracker(res_loss_plt, torch.Tensor([res_running_loss / 30]),torch.Tensor([i + epoch * len(trainloader)]))
            value_tracker(vgg_loss_plt, torch.Tensor([vgg_running_loss / 30]), torch.Tensor([i + epoch * len(trainloader)]))
            vgg_loss_matplt.append([vgg_running_loss])
            res_loss_matplt.append([res_running_loss])
            print('[%d, %5d] vgg loss: %.3f' %(epoch + 1, i + 1, vgg_running_loss / 30))
            print('[%d, %5d] res loss: %.3f' % (epoch + 1, i + 1, res_running_loss / 30))
            res_running_loss = 0.0
            vgg_running_loss = 0.0
    acc1 = acc_check(resnet50, testloader,  res_acc_matplt)
    acc2 = acc_check(vgg16, testloader,  vgg_acc_matplt)
    value_tracker(vgg_acc_plt, torch.Tensor([acc2]), torch.Tensor([epoch]))
    value_tracker(res_acc_plt, torch.Tensor([acc1]), torch.Tensor([epoch]))


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = vgg16(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def plot_compare(loss_list1: list, loss_list2: list, ylim=None, title=None) -> None:
    bn = loss_list1
    nn = loss_list2

    plt.figure(figsize=(15, 10))
    plt.plot(bn, label='vgg')
    plt.plot(nn, label='resnet')
    if ylim:
        plt.ylim(ylim)

    if title:
        plt.title(title)
    plt.legend()
    plt.grid('on')
    plt.show()

plot_compare(vgg_acc_matplt, res_acc_matplt,[0, 1.0], title='Training acc at Epoch')
plot_compare(vgg_loss_matplt, res_loss_matplt, title='Training loss at Epoch')