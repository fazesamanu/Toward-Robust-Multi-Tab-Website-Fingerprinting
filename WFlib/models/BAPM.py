import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class AttentionModule(nn.Module):
    def __init__(self, output_dim):
        super(AttentionModule, self).__init__()
        self.output_dim = output_dim
        self.kernel = None

    def build(self, input_shape, device):
        self.kernel = nn.Parameter(torch.empty(3, input_shape[-1], self.output_dim, device=device))
        nn.init.uniform_(self.kernel)

    def forward(self, x):
        if self.kernel is None:
            self.build(x.shape, x.device)
        wq, wk, wv = [torch.matmul(x, self.kernel[i]) for i in range(3)]
        qk = torch.bmm(wq, wk.transpose(1, 2))
        qk = qk / (self.output_dim ** 0.5)
        qk = F.softmax(qk, dim=-1)
        v = torch.bmm(qk, wv)
        return v


class BAPM(nn.Module):
    """
    Block Attention Profiling Model (BAPM) adapted for multi-tab WF (Section V-A).
    Fuses predictions across all tabs via mean pooling.
    """

    def __init__(self, num_classes=100, num_tab=2):
        super(BAPM, self).__init__()
        filters_num = [32, 64, 128]
        kernels_size = [5, 5, 5]
        pool_size = [8, 8, 8]
        self.time_step = 16
        self.num_tab = max(1, num_tab)

        self.cnn_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=filters_num[0], kernel_size=kernels_size[0], padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(filters_num[0]),
            nn.MaxPool1d(kernel_size=pool_size[0], padding=0, ceil_mode=True),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=filters_num[0], out_channels=filters_num[1], kernel_size=kernels_size[1], padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(filters_num[1]),
            nn.MaxPool1d(kernel_size=pool_size[1], padding=0, ceil_mode=True),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=filters_num[1], out_channels=filters_num[2], kernel_size=kernels_size[2], padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(filters_num[2]),
            nn.MaxPool1d(kernel_size=pool_size[2], padding=0, ceil_mode=True),
            nn.AdaptiveAvgPool1d(self.time_step),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.2),
        )

        self.one_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.TransformerEncoderLayer(d_model=128, nhead=1, dim_feedforward=256, batch_first=True),
                    nn.Flatten(start_dim=1),
                    nn.Dropout(p=0.2),
                )
                for _ in range(self.num_tab)
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.cnn_layer(x)
        x = x.view(x.shape[0], self.time_step, -1)

        multi_head_attn = [self.one_heads[idx](x) for idx in range(self.num_tab)]
        multi_head_attn = [torch.unsqueeze(t, dim=1) for t in multi_head_attn]
        x = torch.cat(multi_head_attn, dim=1)

        x_reshaped = x.contiguous().view(-1, x.size(-1))
        x_reshaped = self.fc(x_reshaped)
        x = x_reshaped.contiguous().view(x.size(0), -1, x_reshaped.size(-1))
        return x.mean(1)