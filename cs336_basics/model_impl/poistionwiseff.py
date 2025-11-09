import torch

class PostionwiseFeedForwardImpl:
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        # d_model: final dimension of the input
        # d_ff: dimension of the hidden layer in the feed-forward network
        # device: Device to store the parameters 
        # dtype: Data type of the parameters
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        # Initialize weights for the three linear transformations
        self.W1 = torch.nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.W2 = torch.nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.W3 = torch.nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))

        # Initialize weights with truncated normal distribution
        stddev1 = (2 / (d_model + d_ff)) ** 0.5
        stddev2 = (2 / (d_ff + d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.W1, mean=0.0, std=stddev1, a=-3*stddev1, b=3*stddev1)
        torch.nn.init.trunc_normal_(self.W2, mean=0.0, std=stddev2, a=-3*stddev2, b=3*stddev2)
        torch.nn.init.trunc_normal_(self.W3, mean=0.0, std=stddev1, a=-3*stddev1, b=3*stddev1)

        self.W1.to(self.device)
        self.W2.to(self.device)
        self.W3.to(self.device)

    def set_weights(self, W1: torch.Tensor, W2: torch.Tensor, W3: torch.Tensor) -> None:
        # W1: A tensor of shape (d_ff, d_model)
        # W2: A tensor of shape (d_model, d_ff)
        # W3: A tensor of shape (d_ff, d_model)
        if W1.shape != (self.d_ff, self.d_model):
            raise ValueError(f"W1 shape mismatch. Expected ({self.d_ff}, {self.d_model}), got {W1.shape}")
        if W2.shape != (self.d_model, self.d_ff):
            raise ValueError(f"W2 shape mismatch. Expected ({self.d_model}, {self.d_ff}), got {W2.shape}")
        if W3.shape != (self.d_ff, self.d_model):
            raise ValueError(f"W3 shape mismatch. Expected ({self.d_ff}, {self.d_model}), got {W3.shape}")
        with torch.no_grad():
            self.W1.copy_(W1)
            self.W2.copy_(W2)
            self.W3.copy_(W3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)
        # SiLU(x) =x·σ(x) = x·(1/(1+exp(−x)))
        x = x.to(self.device)
        W1x = torch.matmul(x, self.W1.T)
        W3x = torch.matmul(x, self.W3.T)
        SiLU_W1x = W1x * torch.sigmoid(W1x)
        return torch.matmul(SiLU_W1x * W3x, self.W2.T)
    