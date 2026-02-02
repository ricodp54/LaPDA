import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义单层交叉注意力
class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttentionLayer, self).__init__()
        self.scale = hidden_dim ** 0.5

    def forward(self, query, key, value):
        # 计算注意力分数

        attention_scores = torch.bmm(query, key.transpose(1, 2)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        # 加权求和得到输出
        output = torch.bmm(attention_weights, value)
        return output