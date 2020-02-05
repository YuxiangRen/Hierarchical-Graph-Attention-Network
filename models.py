import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import NodeAttentionLayer, SchemaAttentionLayer

class HGAT(nn.Module):
    def __init__(self, tfeat, nfeat_list, nhid, shid, nclass, nd_dropout, se_dropout, alpha, nheads):
        """Dense version of GAT."""
        super(HGAT, self).__init__()
        self.nd_dropout = nd_dropout
        self.se_dropout = se_dropout
        self.nheads = nheads
        self.node_level_attentions = []
        for i in range(len(nfeat_list)):
            self.node_level_attentions.append([NodeAttentionLayer(tfeat, nfeat_list[i], nhid, nd_dropout=nd_dropout, alpha=alpha, concat=True) for _ in range(nheads)])

        for i, node_attentions_type in enumerate(self.node_level_attentions):
            for j, node_attention in enumerate(node_attentions_type):
                self.add_module('attention_path_{}_head_{}'.format(i,j), node_attention)
        self.W = nn.Parameter(torch.zeros(size=(tfeat, nhid*nheads)))
        self.schema_level_attention = SchemaAttentionLayer(nhid*nheads, shid, se_dropout, alpha)
        
        self.linear_layer = nn.Linear(shid, nclass)
        
    def forward(self, x_list, adjs):
        x = x_list[0]
        o_list = []
        for i in range(0,len(x_list)-1):
#            o_x = torch.stack([att(x, x_list[i+1], adjs[i]) for att in self.node_level_attentions[i]]).sum(0) / self.nheads
            o_x = torch.cat([att(x, x_list[i+1], adjs[i]) for att in self.node_level_attentions[i]], dim=1)
            o_list.append(o_x)
        x = torch.mm(x_list[0], self.W)  
        x = F.dropout(x, self.se_dropout, training=self.training)
        
        x = self.schema_level_attention(x, o_list)
        
        x = self.linear_layer(x)
        
#        embeddings = x
        
        return F.log_softmax(x, dim=1)


