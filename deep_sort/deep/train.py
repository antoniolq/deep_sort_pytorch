import argparse
import os
import time
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from model import Net
from collections import OrderedDict

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir", default='data', type=str)
parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--gpu-id", default=0, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--interval", '-i', default=20, type=int)
parser.add_argument('--resume', '-r', action='store_true')
args = parser.parse_args()

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loading
# root = args.data_dir
root = "/mnt/Disk1/qingl/data/Mars"
train_dir = os.path.join(root, "bbox_train")
test_dir = os.path.join(root, "bbox_test")

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((128, 64), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# divide
test_size = 0.3
num_workers = 0
train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
test_split = int(np.floor((test_size) * num_train))
test_idx, train_idx = indices[:test_split], indices[test_split:]
print(len(test_idx), len(train_idx))
test_sampler = SubsetRandomSampler(test_idx)
train_sampler = SubsetRandomSampler(train_idx)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=512, sampler=train_sampler, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(train_data, batch_size=512, sampler=test_sampler, num_workers=num_workers)
# trainloader = torch.utils.data.DataLoader(
#     torchvision.datasets.ImageFolder(train_sampler, transform=transform_train),
#     batch_size=1024,shuffle=True
# )
# testloader = torch.utils.data.DataLoader(
#     torchvision.datasets.ImageFolder(test_sampler, transform=transform_test),
#     batch_size=1024,shuffle=True
# )

num_classes1 = len(trainloader.dataset.classes)
print("train num_classes", num_classes1)
num_classes2 = len(testloader.dataset.classes)
print("test num_classes", num_classes2)
num_classes = max(num_classes1, num_classes2)
# net definition
start_epoch = 0
net = Net(num_classes=num_classes)
net = nn.DataParallel(net, device_ids=[1, 2, 3])
cudnn.benchmark = True
args.resume = True
if args.resume:
    assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    print('Loading from checkpoint/ckpt.t7')
    checkpoint = torch.load("./checkpoint/ckpt.t7")
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    new_state_dict = OrderedDict()
    for k, v in net_dict.items():
        if 'module' not in k:
            k = 'module.' + k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k] = v
    net.load_state_dict(new_state_dict)
    net.load_state_dict(checkpoint, False)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
net.to(device)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
best_acc = 0.


# train function for each epoch
def train(epoch):
    print("\nEpoch : %d" % (epoch + 1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print
        if (idx + 1) % interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100. * (idx + 1) / len(trainloader), end - start, training_loss / interval, correct, total,
                100. * correct / total
            ))
            training_loss = 0.
            start = time.time()

    return train_loss / len(trainloader), 1. - correct / total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)

        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
            100. * (idx + 1) / len(testloader), end - start, test_loss / len(testloader), correct, total,
            100. * correct / total
        ))

    # saving checkpoint
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {
            'net_dict': net.module.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.t7')

    return test_loss / len(testloader), 1. - correct / total


# plot figure
x_epoch = []
record = {'train_loss': [], 'train_err': [], 'test_loss': [], 'test_err': []}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")


# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))


def main():
    for epoch in range(start_epoch, start_epoch + 40):
        train_loss, train_err = train(epoch)
        test_loss, test_err = test(epoch)
        print("end")
        # draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        # if (epoch+1)%20==0:
        #     lr_decay()


if __name__ == '__main__':
    main()