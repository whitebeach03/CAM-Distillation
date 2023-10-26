import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from src.utils import *
from src.model import *

epoch = 50
batch_size = 128
num_class = 2
model_size = 'student'
# model_size = 'teacher'
data_name = 'cifar10'
# data_name = 'mnist'
es = EarlyStopping(patience=5, verbose=1)

def main():
    seed_everything(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setting dataset (MNIST)
    if data_name == 'mnist':
        data_dir = './data/MNIST'
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        trainset = datasets.MNIST(root=data_dir, download=True, train=True, transform=transform)
        testset = datasets.MNIST(root=data_dir, download=True, train=False, transform=transform)
        
        trainset = extract_mnist(trainset, 1, 9)
        testset = extract_mnist(testset, 1, 9)

    elif data_name == 'cifar10':
        data_dir = './data/CIFAR10'
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        trainset = datasets.CIFAR10(root=data_dir, download=True, train=True, transform=transform)
        testset = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=transform)
        
        if num_class == 2:
            trainset = extract_cifar10(trainset, 1, 9)
            testset = extract_cifar10(testset, 1, 9)
            trainset = label_classification(trainset, 1, 9)
            testset = label_classification(testset, 1, 9)
            n_train = 6000
            n_val = 1000
        else:
            n_train = int(len(trainset) * 0.9)
            n_val = len(trainset) - n_train

    n_exceed = len(trainset) - n_train - n_val
    trainset, valset, exceed = random_split(trainset, [n_train, n_val, n_exceed])
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    print('Number of trainset: ' , str(n_train))
    print('Number of valset: ' , str(n_val))

    # setting model
    if model_size == 'teacher':
        model = TeacherModel().to(device)
    elif model_size == 'student':
        model = StudentModel2(3, num_class).to(device)

    # setting loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, [10, 15], gamma=0.1)

    # train
    for t in range(epoch):
        lr = get_lr(optimizer)
        train_acc, train_loss = train(model, optimizer, train_dataloader, loss_fn, device)
        val_acc, val_loss = valid(model, val_dataloader, loss_fn, device)
        print(f"| {t:>6} |  {round(train_loss, 7):>12} | {round(train_acc, 7):>12} | {round(val_loss, 7):>12} | {round(val_acc, 7):>12} | {round(lr, 7):>8} |")
        scheduler.step()
        if save_param(es, val_loss, model, model_size, epoch, batch_size, num_class):
            break

def train(model, optimizer, dataloader, loss_fn, device):
    model.train()
    avg_acc = 0
    avg_loss = 0
    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)
        avg_loss += loss.item()

        preds = logits.argmax(dim=1, keepdim=True)
        avg_acc += preds.eq(labels.view_as(preds)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_acc = 100. * avg_acc / len(dataloader.dataset)
    avg_loss /= len(dataloader)
    return avg_acc, avg_loss

def valid(model, dataloader, loss_fn, device):
    model.eval()
    avg_acc = 0
    avg_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)
            avg_loss += loss.item()

            preds = logits.argmax(dim=1, keepdim=True)
            avg_acc += preds.eq(labels.view_as(preds)).sum().item()
    
    avg_acc = 100. * avg_acc / len(dataloader.dataset)
    avg_loss /= len(dataloader)
    return avg_acc, avg_loss


# サンプルの抽出
def extract_mnist(dataset, idx1, idx2):
    mask = (dataset.targets == idx1) | (dataset.targets == idx2)
    dataset.data = dataset.data[mask]
    dataset.targets = dataset.targets[mask]
    return dataset

def extract_cifar10(dataset, idx1, idx2):
    new_dataset = []
    for i in range(len(dataset)):
        current_data = dataset[i]
        if current_data[1] == idx1:
            new_dataset.append(current_data)
        elif current_data[1] == idx2:
            new_dataset.append(current_data)
    return new_dataset

# ラベルの振り分け
def label_classification(dataset, idx1, idx2):
    new_dataset = []
    for i in range(len(dataset)):
        current_data = dataset[i]
        if current_data[1] == idx1:
            l = list(current_data)
            l[1] = 0
            current_data = tuple(l)
            new_dataset.append(current_data)
        elif current_data[1] == idx2:
            l = list(current_data)
            l[1] = 1
            current_data = tuple(l)
            new_dataset.append(current_data)
    return new_dataset


if __name__ == '__main__':
    main()