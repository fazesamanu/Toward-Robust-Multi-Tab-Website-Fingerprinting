import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


class DropPath(nn.Module):
    """Stochastic Depth / DropPath per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class Mlp(nn.Module):
    """MLP as used in Transformer blocks (Section IV-D, Eq. 8)."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TopMAttention(nn.Module):
    """
    Multi-Head Top-m Self-Attention (Section IV-D, Equations 4-7).
    Keeps only the top-m largest attention weights in each query row to filter out
    irrelevant noise from other tabs and WF defenses.
    """

    def __init__(self, dim, num_heads, dropout, top_m):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.top_m = top_m

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout),
        )
        self.proj_drop = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        k_top = min(self.top_m, N)
        if k_top < N and k_top > 0:
            mask = torch.zeros(B, self.num_heads, N, N, device=q.device, requires_grad=False)
            index = torch.topk(attn, k=k_top, dim=-1, largest=True)[1]
            mask.scatter_(-1, index, 1.0)
            attn = torch.where(mask > 0, attn, torch.full_like(attn, float("-inf")))

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(x)
        return x


class MHSA_Block(nn.Module):
    """Transformer block with Top-m Attention, LayerNorm, MLP, and DropPath (Section IV-D)."""

    def __init__(self, embed_dim, nhead, dim_feedforward, dropout, top_m, drop_path_rate=0.1):
        super().__init__()
        self.attn = TopMAttention(embed_dim, nhead, dropout, top_m)
        self.drop_path = DropPath(drop_path_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=dim_feedforward, act_layer=nn.GELU, drop=dropout)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TopM_MHSA(nn.Module):
    """Website Identification module stacking n MHSA_Blocks (Section IV-D, Figure 6)."""

    def __init__(self, embed_dim, num_heads, num_mhsa_layers, dim_feedforward, dropout, top_m, drop_path_rate=0.1):
        super().__init__()
        self.nets = nn.ModuleList(
            [
                MHSA_Block(embed_dim, num_heads, dim_feedforward, dropout, top_m, drop_path_rate=drop_path_rate)
                for _ in range(num_mhsa_layers)
            ]
        )

    def forward(self, x, pos_embed):
        if pos_embed.shape[1] != x.shape[1]:
            pos_embed = F.interpolate(
                pos_embed.permute(0, 2, 1), size=x.shape[1], mode="linear", align_corners=False
            ).permute(0, 2, 1)
        output = x + pos_embed
        for layer in self.nets:
            output = layer(output)
        return output


class ConvBlock1d(nn.Module):
    """Residual 1D Convolutional block in the Local Profiling module (Section IV-C, Figure 5)."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(ConvBlock1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        self.last_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)


class LocalProfiling(nn.Module):
    """
    CNN-based Local Profiling module in ARES / Trans-WF (Section IV-C, Figure 5, Table I).
    Contains L=4 blocks, each with ConvBlock1d (kernel_size=7), MaxPool1d (pool_size=8, stride=4),
    and Dropout(0.1), outputting 256-dim local feature vectors.
    """

    def __init__(self, in_channels=8, num_blocks=4, kernel_size=7, pool_size=8, pool_stride=4, out_dim=256, dropout=0.1):
        super(LocalProfiling, self).__init__()
        channels = [in_channels, 32, 64, 128, out_dim]
        if num_blocks != 4:
            step = [in_channels] + [min(out_dim, 32 * (2 ** i)) for i in range(num_blocks - 1)] + [out_dim]
            channels = step

        layers = []
        for i in range(num_blocks):
            layers.append(ConvBlock1d(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=kernel_size))
            layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride))
            layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ARES(nn.Module):
    """
    ARES: Parameter-shared Trans-WF backbone with N linear classification heads
    for multi-label multi-tab website fingerprinting (Section IV, Figure 2, Table I).
    """

    def __init__(
        self,
        num_classes=100,
        in_channels=8,
        embed_dim=256,
        num_heads=8,
        num_mhsa_layers=4,
        top_m=20,
        dropout=0.1,
        max_len=29,
        num_blocks=4,
        kernel_size=7,
        pool_size=8,
        use_local_profiling=True,
        use_website_id=True,
    ):
        super(ARES, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.use_local_profiling = use_local_profiling
        self.use_website_id = use_website_id

        dim_feedforward = embed_dim * 4

        if use_local_profiling:
            self.profiling = LocalProfiling(
                in_channels=in_channels,
                num_blocks=num_blocks,
                kernel_size=kernel_size,
                pool_size=pool_size,
                out_dim=embed_dim,
                dropout=dropout,
            )
        else:
            # Ablation study (Table VIII): without CNN Local Profiling module
            self.profiling = nn.Sequential(
                nn.AvgPool1d(kernel_size=64, stride=64),
                nn.Conv1d(in_channels, embed_dim, kernel_size=1),
                nn.LayerNorm([embed_dim]),
            )

        self.pos_embed = nn.Embedding(max_len, embed_dim).weight

        if use_website_id:
            self.topm_mhsa = TopM_MHSA(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_mhsa_layers=num_mhsa_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                top_m=top_m,
            )
        else:
            # Ablation study (Table VIII): replace Website Identification module with MLP
            self.topm_mhsa = None
            self.ablation_mlp = nn.Sequential(
                nn.LayerNorm(embed_dim),
                Mlp(in_features=embed_dim, hidden_features=dim_feedforward, act_layer=nn.GELU, drop=dropout),
            )

        # N linear-layer-based heads (Section IV-A, Figure 2)
        self.mlp = nn.Linear(embed_dim, num_classes)

    def extract_features(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if not self.use_local_profiling:
            # AvgPool1d + 1x1 projection without CNN local pattern profiling
            x = self.profiling[0](x)
            x = self.profiling[1](x)
        else:
            x = self.profiling(x)
        x = x.permute(0, 2, 1)
        if self.use_website_id:
            x = self.topm_mhsa(x, self.pos_embed.unsqueeze(0))
        else:
            x = self.ablation_mlp(x)
        x = x.mean(dim=1)
        return x

    def forward(self, x):
        feat = self.extract_features(x)
        out = self.mlp(feat)
        return out

    def expand_monitored_websites(self, new_total_classes):
        """
        Incremental expansion of monitored websites (Section V-E):
        Freezes the shared Trans-WF backbone and appends new site-specific linear heads
        while preserving existing linear head weights.
        """
        assert new_total_classes > self.num_classes, "new_total_classes must exceed current num_classes"
        for param in self.profiling.parameters():
            param.requires_grad = False
        self.pos_embed.requires_grad = False
        if self.topm_mhsa is not None:
            for param in self.topm_mhsa.parameters():
                param.requires_grad = False

        old_linear = self.mlp
        device = old_linear.weight.device
        new_linear = nn.Linear(self.embed_dim, new_total_classes).to(device)
        with torch.no_grad():
            new_linear.weight[: self.num_classes].copy_(old_linear.weight)
            new_linear.bias[: self.num_classes].copy_(old_linear.bias)
            trunc_normal_(new_linear.weight[self.num_classes :], std=0.02)
            nn.init.constant_(new_linear.bias[self.num_classes :], 0.0)

        self.mlp = new_linear
        self.num_classes = new_total_classes
        return self


# Alias for paper terminology
TransWF = ARES


if __name__ == "__main__":
    feat_len = 8000
    in_channels = 8
    net = ARES(num_classes=100)
    x = torch.randn(4, in_channels, feat_len)
    out = net(x)
    print(f"in:{x.shape} --> out:{out.shape}")