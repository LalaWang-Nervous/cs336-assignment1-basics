import torch

class LinearImpl(torch.nn.Module):
    def __init__(self, in_features : int, out_features : int, device=None, dtype=None):
        # in_features: final dimension of the input
        # out_features: final dimension of the output
        # device: Device to store the parameters 
        # ondtype: Data type of the parameters
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # 初始随机化为均值为0，方差为 2/(in_features + out_features), 只取到(-3σ, 3σ)范围内的值
        stddev = (2 / (in_features + out_features)) ** 0.5
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=stddev, a=-3*stddev, b=3*stddev)

        self.weight.to(self.device)

    def set_weights(self, weight: torch.Tensor) -> None:
        # weight: A tensor of shape (out_features, in_features)
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(f"Weight shape mismatch. Expected ({self.out_features}, {self.in_features}), got {weight.shape}")
        with torch.no_grad():
            self.weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return torch.matmul(x, self.weight.T)