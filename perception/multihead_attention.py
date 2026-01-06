import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制实现，适用于图文识别任务
    
    Args:
        d_model (int): 模型维度
        n_heads (int): 注意力头的数量
        dropout (float): Dropout概率
        bias (bool): 是否使用偏置
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, bias: bool = True):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 确保维度可以被头数整除
        assert self.d_k * n_heads == d_model, "d_model必须能被n_heads整除"
        
        # Q, K, V 的线性变换层
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        
        # 输出线性变换层
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        Args:
            query (Tensor): 查询张量，形状为 (batch_size, seq_len_q, d_model)
            key (Tensor): 键张量，形状为 (batch_size, seq_len_k, d_model)
            value (Tensor): 值张量，形状为 (batch_size, seq_len_v, d_model)
            mask (Tensor, optional): 掩码张量，形状为 (batch_size, seq_len_q, seq_len_k)
            
        Returns:
            Tensor: 注意力输出，形状为 (batch_size, seq_len_q, d_model)
        """
        batch_size = query.size(0)
        
        # 线性变换
        Q = self.w_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.w_k(key)    # (batch_size, seq_len_k, d_model)
        V = self.w_v(value)  # (batch_size, seq_len_v, d_model)
        
        # 分割成多头
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len_q, d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len_k, d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len_v, d_k)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, n_heads, seq_len_q, seq_len_k)
        
        # 应用mask
        if mask is not None:
            # mask应该与scores形状相同
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 应用softmax
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, n_heads, seq_len_q, seq_len_k)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重到V
        out = torch.matmul(attention_weights, V)  # (batch_size, n_heads, seq_len_q, d_k)
        
        # 合并所有头
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)  # (batch_size, seq_len_q, d_model)
        
        # 最终线性变换
        out = self.out_proj(out)  # (batch_size, seq_len_q, d_model)
        
        return out

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    # 创建MultiHeadAttention层
    attn = MultiHeadAttention(d_model, n_heads)
    
    # 创建示例输入
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # 创建mask（可选）
    mask = torch.ones(batch_size, seq_len, seq_len)  # 全1的mask
    
    # 前向传播
    output = attn(query, key, value, mask)
    
    print(f"输入形状: query={query.shape}, key={key.shape}, value={value.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试没有mask的情况
    output_no_mask = attn(query, key, value)
    print(f"无mask输出形状: {output_no_mask.shape}")
