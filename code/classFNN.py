# =======================
# Imported libraries
# =======================
import torch.nn as nn
# =======================


# =======================
# FNN Class 
# =======================
class FNN(nn.Module):
    """Feedforward Neural Network (FNN) implementation.
    
    A customizable fully connected neural network with optional dropout and batch normalization.
    The network architecture is defined by the dims parameter which specifies layer sizes.

    Parameters
    ----------
    dims : list of int
        List specifying the number of neurons in each layer.
        First element is input dimension, last is output dimension.
    dropout : float, optional
        Dropout probability (0 means no dropout, default: 0)
    bn : bool, optional
        Whether to use batch normalization (default: False)
    """
    def __init__(self, dims, dropout=0, bn=False):
        super(FNN, self).__init__()
        layers = []
        prev_dim = dims[0]
            
        for dim in dims[1:len(dims)-1]:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            if dropout != 0: 
                layers.append(nn.Dropout(dropout))
            if bn:
                layers.append(nn.BatchNorm1d(dim))
            prev_dim = dim
            
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, dims[-1])
        
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
# =======================