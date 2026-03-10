import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeSeparationLoss(nn.Module):
    def __init__(self, feat_dim, num_classes=2, momentum=0.9, margin=0.2, beta=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.margin = margin
        self.beta = beta

        self.register_buffer("prototypes", F.normalize(torch.randn(num_classes, feat_dim), dim=-1))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update_prototypes(self, z, y):
        for cls in y.unique():
            cls_idx = cls.item()
            z_cls = z[y == cls_idx].mean(dim=0)
            z_cls = F.normalize(z_cls, dim=0)

            if not self.initialized[cls_idx]:
                self.prototypes[cls_idx] = z_cls
                self.initialized[cls_idx] = True
            else:
                p = self.prototypes[cls_idx]
                p = self.momentum * p + (1.0 - self.momentum) * z_cls
                self.prototypes[cls_idx] = F.normalize(p, dim=0)

    def forward(self, z, y):
        z = F.normalize(z, dim=-1)

        if self.training:
            with torch.no_grad():
                self.update_prototypes(z.detach(), y)

        proto_y = self.prototypes[y]
        sim_own = (z * proto_y).sum(dim=-1)
        pull = 1.0 - sim_own

        other_y = 1 - y
        proto_other = self.prototypes[other_y]
        sim_other = (z * proto_other).sum(dim=-1)
        push = F.relu(sim_other - self.margin)

        loss = pull.mean() + self.beta * push.mean()
        return loss, pull.mean(), push.mean(), sim_own.mean(), sim_other.mean()
