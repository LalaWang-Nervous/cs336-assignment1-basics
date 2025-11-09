import torch

class RotaryPositionalEmbeddingImpl(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_k = d_k
        self.theta = theta
        
        # 确保d_k是偶数
        assert d_k % 2 == 0, "d_k must be even for Rotary Positional Embedding"
        
        # 预计算频率
        # theta_i = 1 / (theta^(2i/d_k)) for i in [0, 1, ..., d_k//2-1]
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k))
        
        # 预计算所有位置的正弦和余弦值
        t = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)  # (max_seq_len, 1)
        freqs = t * inv_freq.unsqueeze(0)  # (max_seq_len, d_k//2)
        
        # 将正弦和余弦值缓存为缓冲区
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        
        if device is not None:
            self.sin_cached = self.sin_cached.to(device)
            self.cos_cached = self.cos_cached.to(device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embedding to input tensor.
        
        Args:
            x: Tensor of shape (..., seq_len, d_k)
            token_positions: Tensor of shape (..., seq_len) with position indices
            
        Returns:
            Tensor with rotary positional embedding applied, same shape as x
        """
        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum sequence length {self.max_seq_len}")
        
        # 获取输入形状
        original_shape = x.shape
        x_flat = x.view(-1, seq_len, self.d_k)
        pos_flat = token_positions.view(-1, seq_len)
        
        # 获取对应的正弦和余弦值
        sin = self.sin_cached[pos_flat]  # (batch*..., seq_len, d_k//2)
        cos = self.cos_cached[pos_flat]  # (batch*..., seq_len, d_k//2)
        
        # 将x分成两部分：偶数索引和奇数索引
        # 这是RoPE的标准实现方式：对每对连续的元素应用旋转
        x1 = x_flat[..., 0::2]  # 偶数索引: 0, 2, 4, ...
        x2 = x_flat[..., 1::2]  # 奇数索引: 1, 3, 5, ...
        
        # 应用旋转
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # 将结果交织回原始顺序
        result = torch.stack([rotated_x1, rotated_x2], dim=-1)
        result = result.view(*x_flat.shape[:-1], self.d_k)
        
        # 恢复原始形状
        return result.view(*original_shape)