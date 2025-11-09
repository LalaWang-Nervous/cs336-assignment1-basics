import torch

class EmbeddingImpl(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        # num_embeddings: size of the vocabulary
        # embedding_dim: dimension of each embedding vector
        # device: Device to store the parameters 
        # dtype: Data type of the parameters
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        # Initialize embeddings with uniform distribution in range -3, 3
        limit = 3
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.uniform_(self.weight, a=-limit, b=limit)

    def set_weights(self, weight: torch.Tensor) -> None:
        # weight: A tensor of shape (num_embeddings, embedding_dim)
        if weight.shape != (self.num_embeddings, self.embedding_dim):
            raise ValueError(f"Weight shape mismatch. Expected ({self.num_embeddings}, {self.embedding_dim}), got {weight.shape}")
        with torch.no_grad():
            self.weight.copy_(weight)

    def forward(self, token_ids : torch.Tensor) -> torch.Tensor:

        token_ids = token_ids.to(self.device)
        return self.weight[token_ids]