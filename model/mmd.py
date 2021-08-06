 
import torch

def _rbf_kernel(X, Y, sigma=1., wt=1., K_XY_only=False):

    X_transpose=torch.transpose(X,0,1)
    Y_transpose=torch.transpose(Y,0,1)
    XX = torch.matmul(X,X_transpose)
    XY=torch.matmul(X,Y_transpose)
    YY=torch.matmul(Y,Y_transpose) 

    X_sqnorms = torch.diag(XX)
    Y_sqnorms = torch.diag(YY)

    r = lambda x: torch.unsqueeze(x, 0)
    c = lambda x: torch.unsqueeze(x, 1)

    XYsqnorm = torch.maximum(-2 * XY + c(X_sqnorms) + r(Y_sqnorms), 0.)

    gamma = 1 / (2 * sigma**2)
    K_XY = wt * torch.exp(-gamma * XYsqnorm)

    if K_XY_only:
        return K_XY

    XXsqnorm = torch.maximum(-2 * XX + c(X_sqnorms) + r(X_sqnorms), 0.)
    YYsqnorm = torch.maximum(-2 * YY + c(Y_sqnorms) + r(Y_sqnorms), 0.)

    gamma = 1 / (2 * sigma**2)
    K_XX = wt * torch.exp(-gamma * XXsqnorm)
    K_YY = wt * torch.exp(-gamma * YYsqnorm)

    return K_XX, K_XY, K_YY, wt