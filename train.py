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
batch_size = 100
num_class = 10
model_size = 'teacher'
es = EarlyStopping(patience=5, verbose=1)

def main():
    seed_everything(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setting dataset (MNIST)
    data_dir = './data/MNIST'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    trainset = datasets.MNIST(root=data_dir, download=True, train=True, transform=transform)
    testset = datasets.MNIST(root=data_dir, download=True, train=False, transform=transform)

    n_train = int(len(trainset) * 0.8)
    n_val = len(trainset) - n_train
    trainset, valset = random_split(trainset, [n_train, n_val])
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # setting model
    if model_size == 'teacher':
        model = TeacherModel().to(device)
    elif model_size == 'student':
        model = StudentModel().to(device)

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
        print(logits.shape)
        print(labels.shape)
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


if __name__ == '__main__':
    main()