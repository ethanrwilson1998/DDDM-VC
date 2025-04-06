import numpy as np
import torch


def identityDP_anonymize(embedding, epsilon: float = 1, should_normalize: bool = False):
    device, dtype = embedding.device, embedding.dtype
    dims = embedding.shape

    emb = embedding.cpu().detach().numpy()
    emb = np.squeeze(emb)

    if should_normalize:
        emb = emb / np.linalg.norm(emb)

    # Sensitivity computed on VoxCeleb1 test set
    sensitivity = 24.855695724487305

    if epsilon > 0:
        noise = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=emb.shape)
        perturbed = emb + noise
    elif epsilon == 0:
        # If epsilon is 0, return a completely random unit vector.
        perturbed = np.random.randn(emb.shape[0])
    else:
        perturbed = emb

    if should_normalize:
        perturbed = perturbed / np.linalg.norm(perturbed)

    return torch.tensor(perturbed, device=device, dtype=dtype).reshape(dims)
