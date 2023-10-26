import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from torch.utils.data import DataLoader
import cv2
import numpy as np
from src.model import *
from src.utils import *

class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,8,3,1,1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,16,5,1,2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(16,32,3,1,1)
        self.bn32= nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32,10,1)

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(self.bn2(x)))
        x = self.pool(x)
        x = self.relu(self.conv3(self.bn4(x)))
        x = self.pool(x)
        x = self.sig(self.conv4(self.bn32(x)))
        x = self.gap(x) # Global Average Pooling
        x = x.view(-1,10)
        return x



model_size = 'teacher'
teacher = TeacherModel()
student = StudentModel2(3, 2)
epoch = 50
batch_size = 10
num_class = 2
torch.manual_seed(123)
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
# datasets = datasets.MNIST(root='./data/MNIST', download=True, train=True, transform=transform)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
datasets = datasets.CIFAR10(root='./data/CIFAR10', download=True, train=True, transform=transform)

def extract_CAM(model, images, labels, batch_size):
    extractor = create_feature_extractor(model, ['sig'])
    features = extractor(images)['sig']
    cam = np.array([])
    for i in range(batch_size):
        label = labels[i]
        feature = features[i][label] 
        feature = feature.detach().numpy()
        c = cv2.resize(feature, (28, 28))
        cam = np.append(cam, c).reshape(i+1, 28, 28)
    cam = torch.tensor(cam)
    return cam

def extract_sample(dataset):
    mask = (dataset.targets == 0) | (dataset.targets == 6)
    dataset.data = dataset.data[mask]
    dataset.targets = dataset.targets[mask]
    return dataset

def label_classification(dataset):
    new_dataset = []
    for i in range(len(dataset)):
        current_data = dataset[i]
        if current_data[1] == 1:
            l = list(current_data)
            l[1] = 0
            current_data = tuple(l)
            new_dataset.append(current_data)
        elif current_data[1] == 9:
            l = list(current_data)
            l[1] = 1
            current_data = tuple(l)
            new_dataset.append(current_data)
    return new_dataset

def ex_sample(dataset):
    new_dataset = []
    for i in range(len(dataset)):
        current_data = dataset[i]
        if current_data[1] == 1:
            # current_data[1] = 0
            new_dataset.append(current_data)
        elif current_data[1] == 9:
            # current_data[1] = 1
            new_dataset.append(current_data)
    return new_dataset


datasets = ex_sample(datasets)
datasets = label_classification(datasets)
dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)

student.load_state_dict(torch.load('./logs/student/' + str(epoch) + '_' + str(batch_size) + '_' + str(num_class) + '_param.pth'))
student.eval()
for images, labels in datasets:
    cam = extract_CAM(student, images, labels, batch_size)
    
    create_CAM(images, cam, epoch, 'student')
    
    break




