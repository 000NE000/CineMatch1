import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        """
        gamma: focusing parameter, 일반적으로 2.0
        weight: 클래스 가중치 (tensor) - CrossEntropyLoss와 동일하게 사용 가능
        reduction: 'mean' 또는 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (batch, seq_len, num_labels)
        # targets: (batch, seq_len)
        ce_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                                  weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = exp(-loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

