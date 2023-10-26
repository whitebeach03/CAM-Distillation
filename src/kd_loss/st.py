import torch
import torch.nn as nn

class SoftTargetLoss(nn.Module):
    '''
    Distilling the knowledge in a Neural Network
    https://arxiv.org/abs/1503.02531
    '''
    def __init__(self, T=10):
        super().__init__()
        self.T = T
    
    def forward(self, logits, targets, t):
        '''
        logits : output of the student
        targets: output of the teacher
        t      : correct label
        '''
        logits = logits / self.T 
        targets = targets / self.T
        soft_loss = nn.KLDivLoss(reduction='batchmean')
        hard_loss = nn.CrossEntropyLoss()
        p = nn.Softmax(dim=1)
        q = nn.LogSoftmax(dim=1)
        kd_loss = soft_loss(q(logits), p(targets)) + hard_loss(logits, t)
        return kd_loss
        
if __name__ == '__main__':
    logits = torch.randn((32,10))
    targets = torch.randn((32,10))
    t = torch.randint(0, 10, (32,))
    loss = SoftTargetLoss()
    kd_loss = loss(logits, targets, t)
    print(kd_loss)
    print(loss.T)