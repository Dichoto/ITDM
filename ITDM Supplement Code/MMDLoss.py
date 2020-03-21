import torch
import torch.nn as nn
import torch.nn.functional as F

class RBFMMD(nn.Module):

    def __init__(self, sigma=[1,2,4,8,16], use_est_width=True, use_gpu=True):
        """
        Args:
            sigma: bandwidth of multiple gaussian kernel
            est_width: estimate the bandwidth as the distance 
                       between mean vectors of the two minibatches.
                       In this case, sigma becomes the multiplyer of the est_width:
                       bandwidth of kernel = sigma*est_width
        """
        super(RBFMMD, self).__init__()
        self.use_gpu = use_gpu
        self.sigma = sigma
        self.use_est_width = use_est_width
    
    def get_scale_matrix(self, M, N):
        # first 'N' entries have '1/N', next 'M' entries have '-1/M'
        if self.use_gpu:
            device = "cuda"
        else:
            device = "cpu"
        s1 = (torch.ones((N, 1)) * 1.0 / N).to(device)
        s2 = (torch.ones((M, 1)) * -1.0 / M).to(device)
        return torch.cat((s1, s2), 0)

    def forward(self, mb_x1, mb_x2):
        """
        Args:
            mb_x1: the first mini-batch of input X (or its embedding as the last hidden layer)
            mb_x2: the second mini-batch of input X (or its embbeding)
        """
        X = torch.cat((mb_x1,mb_x2), dim=0) #concatenate mini-batch
        XX = torch.matmul(X, X.t())
        X2 = torch.sum(X*X, 1, keepdim=True)
        exp = 2* XX - X2 - X2.t()
        M = mb_x1.size()[0]
        N = mb_x2.size()[0]
        s = self.get_scale_matrix(M, N)
        S = torch.matmul(s, s.t())

        loss = 0
        
        est_width = 1 # use estimated width between two minibatches, use the median (approxiamtely)
        if self.use_est_width:
            est_width = -1 * torch.median(exp).item()
        for v in self.sigma:
            kernel_val = torch.exp(exp / (v*est_width))
            loss += torch.sum(S * kernel_val)       
        loss = loss/len(self.sigma)
        loss = torch.sqrt(loss)
        return loss
