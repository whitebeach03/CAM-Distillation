import cv2
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
from torchvision.models.feature_extraction import create_feature_extractor

from src.utils import *
from src.model import *
from src.kd_loss.st import SoftTargetLoss
from src.kd_loss.CAM_loss import CAMTargetLoss

epoch = 10
batch_size = 100
num_class = 10
kd = 'cam'

def main():
    seed_everything(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setting dataset (MNIST)
    data_dir = './data/MNIST'
    transform = transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    trainset = datasets.MNIST(root=data_dir, download=True, train=True, transform=transform)
    testset = datasets.MNIST(root=data_dir, download=True, train=False, transform=transform)

    n_train = int(len(trainset) * 0.8)
    n_val = len(trainset) - n_train
    trainset, valset = random_split(trainset, [n_train, n_val])
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # setting model 
    teacher = TeacherModel().to(device)
    student = StudentModel().to(device)

    # setting loss function 
    if kd == 'st':
        loss_fn = nn.CrossEntropyLoss()
        kd_loss_fn = SoftTargetLoss()
        optimizer = Adam(student.parameters())
    elif kd == 'cam':
        loss_fn = nn.CrossEntropyLoss()
        kd_loss_fn = CAMTargetLoss()
        optimizer = Adam(student.parameters())

    # training step
    if kd == 'st':
        print('Distillation of Hinton')
        teacher.load_state_dict(torch.load('./logs/teacher/' + str(epoch) + '_' + str(batch_size) + '_' + str(num_class) + '_param.pth'))
        for t in range(epoch):
            train_acc, train_loss = train_kd(student, teacher, optimizer, train_dataloader, kd_loss_fn, device)
            val_acc, val_loss = valid(student, val_dataloader, loss_fn,  device)
            print(f"| {t:>6} |  {round(train_loss, 7):>12} | {round(train_acc, 7):>12} | {round(val_loss, 7):>12} | {round(val_acc, 7):>12} |")
    elif kd == 'cam':
        print('CAM-based Distillation')
        teacher.load_state_dict(torch.load('./logs/teacher/' + str(epoch) + '_' + str(batch_size) + '_' + str(num_class) + '_param.pth'))
        for t in range(epoch):
            train_acc, train_loss = CAM_kd(student, teacher, optimizer, train_dataloader, kd_loss_fn, device)
            val_acc, val_loss = valid(student, val_dataloader, loss_fn, device)
            print(f"| {t:>6} |  {round(train_loss, 7):>12} | {round(train_acc, 7):>12} | {round(val_loss, 7):>12} | {round(val_acc, 7):>12} |")

def train_kd(student, teacher, optimizer, dataloader, loss_fn, device):
    student.train()
    teacher.eval()
    avg_acc = 0
    avg_loss = 0
    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = student(images)  # torch.Size([batch_size=100, 10])
        targets = teacher(images) # torch.Size([batch_size=100, 10])
        
        loss = loss_fn(logits, targets, labels) 
        avg_loss += loss.item()
        
        preds = logits.argmax(dim=1, keepdim=True)
        avg_acc += preds.eq(labels.view_as(preds)).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_acc = 100. * avg_acc / len(dataloader.dataset)
    avg_loss /= len(dataloader)
    return avg_acc, avg_loss

def CAM_kd(student, teacher, optimizer, dataloader, loss_fn, device):
    student.train()
    teacher.eval()
    avg_acc = 0
    avg_loss = 0
    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = student(images)  # torch.Size([batch_size=100, 10])
        targets = teacher(images) # torch.Size([batch_size=100, 10])

        # CAM情報の抽出 
        student_cam = extract_CAM(student, images, labels, batch_size, device)
        teacher_cam = extract_CAM(teacher, images, labels, batch_size, device) # torch.Size([batch_size=100, 28, 28])

        loss = loss_fn(batch_size, logits, targets, student_cam, teacher_cam, labels)
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
        for images, labels in tqdm(dataloader, leave=False):
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

def extract_CAM(model, images, labels, batch_size, device):
    extractor = create_feature_extractor(model, ['layer4'])
    features = extractor(images)['layer4']
    cam = np.array([])

    for i in range(batch_size):
        label = labels[i]
        feature = features[i][label].to(device)
        feature = feature.detach().cpu().numpy() # numpyに変換
        c = cv2.resize(feature, (32, 32))
        cam = np.append(cam, c).reshape(i+1, 32, 32)

    cam = torch.tensor(cam)
    return cam



if __name__ == '__main__':
    main()
