import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import ipdb
import json
# import clip
import skimage.io as io
from PIL import Image
import ipdb
from transformers.activations import NewGELUActivation
import math

'''
class Selector_Mlp(nn.Module):
    
    def __init__(self):
        super(Selector_Mlp, self).__init__()
        
        self.gpt_embedding_size = 768
        #self.head = nn.Linear(self.gpt_embedding_size, 1)
        self.head = nn.Sequential(
            nn.Linear(self.gpt_embedding_size, self.gpt_embedding_size // 2),
            #NewGELUActivation(),
            nn.ReLU(),
            nn.Linear(self.gpt_embedding_size // 2, 1),
        )

        # nn.init.kaiming_uniform_(self.head[0].weight, a=math.sqrt(5))
        # nn.init.zeros_(self.head[2].weight)
        # nn.init.zeros_(self.head[0].bias)
        # nn.init.zeros_(self.head[2].bias)
        

    def forward(self, embeddings: torch.Tensor):
        #eos_embedding = torch.stack(eos_embedding)    # (B, C)
        embeddings = embeddings.to(dtype=self.head[0].weight.dtype)
        score = self.head(embeddings)  # (B, 1)
        score = score.squeeze(dim=1).softmax(-1)
        #ipdb.set_trace()
        return score
'''


class TransformerSelector(nn.Module):
    def __init__(self, embedding_dim=768, num_layers=4, num_heads=8, dim_feedforward=1536, dropout=0.1):
        """
        基于 Transformer 的选择模型，从输入的 3 个嵌入中挑选一个。

        Args:
            embedding_dim (int): 输入嵌入的维度。
            num_layers (int): Transformer 层的数量。
            num_heads (int): 多头注意力机制中的头数。
            dim_feedforward (int): 前馈网络的隐藏层维度。
            dropout (float): Dropout 概率。
        """
        super(TransformerSelector, self).__init__()

        # Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头，用于选择哪个 embedding
        self.head = nn.Linear(embedding_dim, 1)  # 输出单个值作为分数

    def forward(self, inputs):
        """
        前向传播。

        Args:
            inputs (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, embedding_dim)。
        
        Returns:
            torch.Tensor: 每个 embedding 的分数，形状为 (batch_size, seq_len)。
        """
        # Transformer Encoder expects (seq_len, batch_size, embedding_dim)
        #ipdb.set_trace()
        inputs = inputs.permute(1, 0, 2).to(self.head.weight.dtype)  # 转换为 (seq_len, batch_size, embedding_dim)
        transformer_output = self.transformer_encoder(inputs)  # 输出形状: (seq_len, batch_size, embedding_dim)

        # 逐个 embedding 通过分类头，得到分数
        logits = self.head(transformer_output)  # 形状: (seq_len, batch_size, 1)
        logits = logits.squeeze(-1).permute(1, 0)  # 转换为 (batch_size, seq_len)
        logits = logits.softmax(-1)

        return logits
