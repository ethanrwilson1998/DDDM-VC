import numpy as np
import torch

def anonymize_voice(embedding, epsilon: float = 1):
    
    device, dtype = embedding.device, embedding.dtype
    dims = embedding.shape
    
    emb = embedding.cpu().detach().numpy()
    emb = np.squeeze(emb)
    
    emb = emb / np.linalg.norm(emb)
    
    # Sensitivity
    sensitivity = 24.855695724487305
    
    if epsilon > 0:
        noise = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=emb.shape)
        perturbed = emb + noise
    elif epsilon == 0:
        # If epsilon is 0, return a completely random unit vector.
        perturbed = np.random.randn(emb.shape[0])
    else:
        perturbed = emb
    
    # perturbed = perturbed / np.linalg.norm(perturbed)
    return torch.tensor(perturbed, device=device, dtype=dtype).reshape(dims)
