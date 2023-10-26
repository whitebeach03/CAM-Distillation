# CAM-based distillationの損失関数の定義

import torch
import torch.nn as nn

class CAMTargetLoss(nn.Module):
    def __init__(self, T=10):
        super().__init__()
        self.T = T
    
    def forward(self, batch_size, logits, targets, student_cam, teacher_cam, t):
        '''
        logits : output of the student
        targets: output of the teacher
        student_cam: CAM information of the student
        teacher_cam: CAM information of the teacher
        t      : correct label
        '''
        lambda_hard = 0.3
        lambda_cam = 0.2

        logits = logits / self.T 
        targets = targets / self.T 
        student_cam = torch.reshape(student_cam, (batch_size, 1, 28*28))
        teacher_cam = torch.reshape(teacher_cam, (batch_size, 1, 28*28))
        soft_loss = nn.KLDivLoss(reduction='batchmean') # E_soft
        hard_loss = nn.CrossEntropyLoss()               # E_hard
        CAM_loss = nn.MSELoss()                         # E_CAM

        p = nn.Softmax(dim=1)
        q = nn.LogSoftmax(dim=1)

        kd_loss = (1-lambda_hard-lambda_cam) * soft_loss(q(logits), p(targets)) + lambda_hard * hard_loss(logits, t) + lambda_cam * CAM_loss(student_cam, teacher_cam)

        return kd_loss

