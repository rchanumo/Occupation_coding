import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class multiclassifier(nn.Module):
 
    def __init__(self, feature_dim, n_classes_list, dropout=0.2):
    	super().__init__()
    	self.n_classes_list = n_classes_list
    	self.linears = nn.ModuleList([nn.Linear(feature_dim, n_classes) for n_classes in n_classes_list])
    	self.dropout = nn.Dropout(dropout)

    def forward(self, x):
    	x = self.dropout(x)
    	outs = []
    	for i in range(len(self.n_classes_list)):
    		outs.append(self.linears[i](x))
    	return outs
                