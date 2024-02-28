import torch
import torch.nn as nn
from einops import repeat

class SinkhornDistance(nn.Module):
    def __init__(self, eps, max_iter, metric:str ='euclidean'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.metric = metric

    def forward(self, x, y, r):
        
        if self.metric == 'euclidean':
            # Calculate the Euclidean distance matrix
            C = torch.sqrt(torch.sum((x.reshape(x.shape[0],1,x.shape[1])-y)**2, dim=-1))
        elif self.metric == 'cosine':
            # Calculate the 1 - cosine similarity matrix
            a=torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
            b = torch.sqrt(torch.sum(y**2,dim=1,keepdim=True))
            C = torch.mm(x, y.T) / torch.mm(a, b.T)
            C = 1-C
        else:
            raise ValueError("Invalid metric. Supported metrics are 'euclidean' and 'cosine'.")

        K = torch.exp(-C / self.eps)

        # Initialize u and v for each sample in the batch
        u = torch.ones(x.shape[0],1).to(x.device)
        v = torch.ones_like(r).to(x.device)

        thresh = 1e-1

        # Sinkhorn algorithm iterations
        for _ in range(self.max_iter):
            # ipdb.set_trace()
            u_prev = u
            u = 1 / (torch.sum(K*v, dim=-1).reshape(-1,1) + 1e-8)
            v = r / (repeat(u.reshape(-1), "s-> s l", l = K.shape[1]) + 1e-8)

            err = (u - u_prev).abs().sum(-1).mean()

            # Check convergence
            if err.item() < thresh:
                break

        # P = torch.diag(u) @ K @ torch.diag(v)
        P = repeat(u.reshape(-1),  "s-> s l", l = K.shape[1]) * K * v

        # Compute Sinkhorn distance for each sample in the batch
        distance = torch.sum(P * C, dim=-1)

        return distance
