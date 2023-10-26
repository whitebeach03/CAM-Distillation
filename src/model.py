import torch
import torch.nn as nn

class TeacherConvBnAct(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, 3, padding=1)
        self.dropout = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class StudentConvBnAct(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, 3, padding=1)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def _make_teacher_layer(self, n, input_dim, output_dim):
        layers = []
        for _ in range(n):
            layers.append(TeacherConvBnAct(input_dim, output_dim))
            input_dim = output_dim
        return nn.Sequential(*layers)
    
    def _make_student_layer(self, n, input_dim, output_dim):
        layers = []
        for _ in range(n):
            layers.append(StudentConvBnAct(input_dim, output_dim))
            input_dim = output_dim
        return nn.Sequential(*layers)

class StudentModel(BaseModel):
    def __init__(self, input_dim=1, output_dim=10):
        super().__init__()
        self.layer1 = self._make_student_layer(4, input_dim, 16)
        self.layer2 = self._make_student_layer(4, 16, 32)
        self.layer3 = self._make_student_layer(4, 32, 64)
        self.layer4 = self._make_student_layer(4, 64, 128)
        self.mlp = MLP(128, 128, output_dim)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        # self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.fc = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)

        x = self.gap(x)
        x = x.view(-1, 10)
        return x

class TeacherModel(BaseModel):
    def __init__(self, input_dim=3, output_dim=2):
        super().__init__()
        self.output_dim = output_dim
        self.layer1 = self._make_teacher_layer(4, input_dim, 16)
        self.layer2 = self._make_teacher_layer(4, 16, 32)
        self.layer3 = self._make_teacher_layer(4, 32, 64)
        self.layer4 = self._make_teacher_layer(4, 64, 128)
        self.mlp = MLP(128, 128, output_dim)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        self.cam = x.detach().numpy() # extract activation map 
        x = self.gap(x)
        x = x.view(x.shape[0], -1)
        # x = self.mlp(x)
        x = self.fc(x)
        return x
    
    def return_CAM(self):
        return self.cam

class StudentModel2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(input_dim, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn32 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(self.bn2(x)))
        x = self.pool(x)
        x = self.relu(self.conv3(self.bn4(x)))
        x = self.pool(x)
        x = self.sig(self.conv4(self.bn32(x)))
        self.cam = x.detach().cpu().numpy()
        x = self.gap(x) # Global Average Pooling
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def return_CAM(self):
        return self.cam

class TeacherModel2(nn.Module):
    pass


# モデルのパラメータ数のカウント
teacher = TeacherModel()
student = StudentModel2(1, 10)

def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

print(f"Teacher parameters: {count_parameters(teacher)}")
print(f"Student parameters: {count_parameters(student)}")

# teacher parameters: 128,610
# student parameters:  48.492