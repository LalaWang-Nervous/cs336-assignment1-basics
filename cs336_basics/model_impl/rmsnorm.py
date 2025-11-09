import torch

class RmsNormImpl:
    def __init__(self, d_model : int, eps : float=1e-5, device = None, dtype = None):
        # d_model: final dimension of the input
        # eps: a small value to avoid division by zero
        # device: Device to store the parameters 
        # dtype: Data type of the parameters
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.gain = torch.nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def set_weights(self, gain: torch.Tensor) -> None:
        # gain: A tensor of shape (d_model,)
        if gain.shape != (self.d_model,):
            raise ValueError(f"Gain shape mismatch. Expected ({self.d_model},), got {gain.shape}")
        with torch.no_grad():
            self.gain.copy_(gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape(batch_size, sequence_length, d_model)and return a tensor of the same shape
        x = x.to(self.device)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return x_normalized * self.gain