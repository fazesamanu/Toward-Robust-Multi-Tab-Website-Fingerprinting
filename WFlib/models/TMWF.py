import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(embed_dim, nhead, dim_feedforward, dropout),
            num_encoder_layers,
        )
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(embed_dim, nhead, dim_feedforward, dropout),
            num_decoder_layers,
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):
        bs = src.shape[0]
        if pos_embed.shape[1] != src.shape[1]:
            pos_embed = F.interpolate(
                pos_embed.permute(0, 2, 1), size=src.shape[1], mode="linear", align_corners=False
            ).permute(0, 2, 1)
        memory = self.encoder(src, pos_embed)
        query_embed = query_embed.repeat(bs, 1, 1)
        tgt = torch.zeros_like(query_embed)
        output = self.decoder(tgt, memory, pos_embed, query_embed)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos, query_pos):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, pos, query_pos)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, src, pos):
        src2 = self.self_attn(query=src + pos, key=src + pos, value=src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, tgt, memory, pos, query_pos):
        tgt2 = self.self_attn(query=tgt + query_pos, key=tgt + query_pos, value=tgt)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt + query_pos, key=memory + pos, value=memory)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class DFNet(nn.Module):
    def __init__(self, dropout):
        super(DFNet, self).__init__()
        filter_num = [0, 32, 64, 128, 256]
        kernel_size = [0, 8, 8, 8, 8]
        conv_stride_size = [0, 1, 1, 1, 1]
        pool_stride_size = [0, 4, 4, 4, 4]
        pool_size = [0, 8, 8, 8, 8]

        self.block1_conv1 = nn.Conv1d(1, filter_num[1], kernel_size[1], conv_stride_size[1], kernel_size[1] // 2)
        self.block1_bn1 = nn.BatchNorm1d(filter_num[1])
        self.block1_elu1 = nn.ELU(alpha=1.0)
        self.block1_conv2 = nn.Conv1d(filter_num[1], filter_num[1], kernel_size[1], conv_stride_size[1], kernel_size[1] // 2)
        self.block1_bn2 = nn.BatchNorm1d(filter_num[1])
        self.block1_elu2 = nn.ELU(alpha=1.0)
        self.block1_pool = nn.MaxPool1d(pool_size[1], pool_stride_size[1], pool_size[1] // 2)
        self.block1_dropout = nn.Dropout(p=dropout)

        self.block2_conv1 = nn.Conv1d(filter_num[1], filter_num[2], kernel_size[2], conv_stride_size[2], kernel_size[2] // 2)
        self.block2_bn1 = nn.BatchNorm1d(filter_num[2])
        self.block2_relu1 = nn.ReLU()
        self.block2_conv2 = nn.Conv1d(filter_num[2], filter_num[2], kernel_size[2], conv_stride_size[2], kernel_size[2] // 2)
        self.block2_bn2 = nn.BatchNorm1d(filter_num[2])
        self.block2_relu2 = nn.ReLU()
        self.block2_pool = nn.MaxPool1d(pool_size[2], pool_stride_size[2], pool_size[2] // 2)
        self.block2_dropout = nn.Dropout(p=dropout)

        self.block3_conv1 = nn.Conv1d(filter_num[2], filter_num[3], kernel_size[3], conv_stride_size[3], kernel_size[3] // 2)
        self.block3_bn1 = nn.BatchNorm1d(filter_num[3])
        self.block3_relu1 = nn.ReLU()
        self.block3_conv2 = nn.Conv1d(filter_num[3], filter_num[3], kernel_size[3], conv_stride_size[3], kernel_size[3] // 2)
        self.block3_bn2 = nn.BatchNorm1d(filter_num[3])
        self.block3_relu2 = nn.ReLU()
        self.block3_pool = nn.MaxPool1d(pool_size[3], pool_stride_size[3], pool_size[3] // 2)
        self.block3_dropout = nn.Dropout(p=dropout)

        self.block4_conv1 = nn.Conv1d(filter_num[3], filter_num[4], kernel_size[4], conv_stride_size[4], kernel_size[4] // 2)
        self.block4_bn1 = nn.BatchNorm1d(filter_num[4])
        self.block4_relu1 = nn.ReLU()
        self.block4_conv2 = nn.Conv1d(filter_num[4], filter_num[4], kernel_size[4], conv_stride_size[4], kernel_size[4] // 2)
        self.block4_bn2 = nn.BatchNorm1d(filter_num[4])
        self.block4_relu2 = nn.ReLU()
        self.block4_pool = nn.MaxPool1d(pool_size[4], pool_stride_size[4], pool_size[4] // 2)
        self.block4_dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        x = input.unsqueeze(1) if len(input.shape) == 2 else input
        x = self.block1_dropout(self.block1_pool(self.block1_elu2(self.block1_bn2(self.block1_conv2(self.block1_elu1(self.block1_bn1(self.block1_conv1(x))))))))
        x = self.block2_dropout(self.block2_pool(self.block2_relu2(self.block2_bn2(self.block2_conv2(self.block2_relu1(self.block2_bn1(self.block2_conv1(x))))))))
        x = self.block3_dropout(self.block3_pool(self.block3_relu2(self.block3_bn2(self.block3_conv2(self.block3_relu1(self.block3_bn1(self.block3_conv1(x))))))))
        x = self.block4_dropout(self.block4_pool(self.block4_relu2(self.block4_bn2(self.block4_conv2(self.block4_relu1(self.block4_bn1(self.block4_conv1(x))))))))
        return x.transpose(1, 2)


class TMWF(nn.Module):
    """
    Transformer-based Model for Multi-Tab Website Fingerprinting (TMWF)
    adapted as described in Section V-A (fusing predictions across tabs).
    """

    def __init__(self, num_classes=100, num_tab=2):
        super(TMWF, self).__init__()
        embed_dim = 256
        nhead = 8
        dim_feedforward = 256 * 4
        num_encoder_layers = 2
        num_decoder_layers = 2
        max_len = 121
        num_queries = max(1, num_tab)
        dropout = 0.1

        self.cnn_layer = DFNet(dropout)
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.trm = Transformer(embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.pos_embed = nn.Embedding(max_len, embed_dim).weight
        self.query_embed = nn.Embedding(num_queries, embed_dim).weight
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input):
        x = self.cnn_layer(input)
        feat = self.proj(x)
        o = self.trm(feat, self.query_embed.unsqueeze(0), self.pos_embed.unsqueeze(0))
        logits = self.fc(o)
        return logits.mean(1)
