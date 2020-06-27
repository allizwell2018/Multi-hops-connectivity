

import pdb 
import torch
from torch.nn import Parameter
from torch import nn
# from torch_geometric.nn.inits import glorot, zeros

class wsgcn_multi(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, num_nodes, multi=1, multi_aggre='concat', 
    			featureTrans=True, improved=False, nonsharing=False,initial_gain=0.1,weight_mask=1):
        super(wsgcn_multi, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.featureTrans = featureTrans 
        self.multi = multi
        self.multi_aggre = multi_aggre
        self.nonsharing = nonsharing
        self.initial_gain = initial_gain
        self.weight_mask = weight_mask

        if self.nonsharing:
            self.weight_nonsharing = Parameter(torch.Tensor(num_nodes, num_nodes)) 
            self.bias_nonsharing = Parameter(torch.Tensor(num_nodes,1))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight_nonsharing, self.initial_gain) 
        nn.init.xavier_normal_(self.bias_nonsharing)
     
        with torch.no_grad():
        	# pdb.set_trace()
        	self.weight_nonsharing.mul_(self.weight_mask.to(self.weight_nonsharing.device)) 

    def normalization(self, adj):
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        return adj

    def forward(self, x, adj, weight2, weight, bias, mask=None, add_loop=False, residue=False):
        
        # assert weight2.size(0)==self.multi,'multi dose not match with weight2'

        if x.dim() == 2:
            x = x.unsqueeze(0)
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        else:
            x = x

        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
            adj = adj.unsqueeze(1)
        elif adj.dim() == 3:
            adj = adj.unsqueeze(1)
        else:
            adj = adj

    
        B, C, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, :, idx, idx] = 1 if not self.improved else 2                
        
        if self.multi == 1 :
            if self.nonsharing:
                adj = adj * self.weight_nonsharing
            else:
                adj = adj * weight2       
        
        
        if self.multi == 1 :
            if residue:            
                out = torch.matmul(adj, x) + x  ## 引入residue连接的思想
            else:
                out = torch.matmul(adj, x)  ## result:32*3*246*1   x: 32*1*246*1 or 1*1*246*1  好像更好
        else:
            if residue: 
                out = torch.matmul(adj, x * weight2) + x  ## 引入residue连接的思想
            else:
                out = torch.matmul(adj, x * weight2)  

        if self.multi_aggre=='max':
            out, _ = torch.max(out,1)   
        elif self.multi_aggre=='mean':
            out = torch.mean(out,1)        ## 32*246*(1)
        elif self.multi_aggre=='concat':
            B1, C1, N1, F1 = out.size()
            out = out.permute(0,2,3,1).view(B1,N1,F1*C1)    ## 32*246*(1*3)

        if self.featureTrans:
           
            out = out * weight
            out = torch.mean(out, -1, keepdim=True)   ## 32*246*1
    
        
        if self.nonsharing:
            out = out + self.bias_nonsharing
        else:
            out = out + bias     

        
        if mask is not None:
            out = out * mask.view(B, C, N, 1).to(x.dtype)

        return out