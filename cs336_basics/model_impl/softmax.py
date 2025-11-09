import torch

class SoftmaxImpl(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    # TODO: 这里似乎是只对第dim维度的才做softmax
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # our function should take two parameters: a tensor and a dimension i, and apply softmax to the i-th dimension of the input tensor. 
        # The output tensor should have the same shape as the input tensor, 
        # but its i-th dimension will now have a normalized probability distribution
        # do not use torch.nn.functional.softmax 
        # only do softmax on the dim dimension
        logits = logits.to(torch.float32)  # Ensure logits are in float32 for numerical stability
        max_logits, _ = torch.max(logits, dim=self.dim, keepdim=True)
        exp_logits = torch.exp(logits - max_logits)  # Subtract max for numerical stability
        sum_exp = torch.sum(exp_logits, dim=self.dim, keepdim=True)
        softmax_output = exp_logits / sum_exp
        return softmax_output.to(logits.dtype)  # Convert back to original dtype if necessary