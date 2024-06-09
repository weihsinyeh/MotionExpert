import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class dim_conv(nn.Module):

    def __init__(self, alignment=True):
        super().__init__()
        self.alig = alignment
        self.embedding = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,768)
        )
        
    '''
    # @ x : input tensor
    # x is the output embedding of alignment network 
    # dimension is [ batchsize, vertex(22), seq_length (T), channel (512)] 
    # @ return : batchsize, vertex(22), 768
    '''
    def forward(self, x):
        '''   
        # if alignment the input x is [ batchsize, vertex ,seq_length, channel]
        # if not alignment the input x is [ batchsize, seq_length, vertex, channel]
        '''
        B,T,V,C = x.size()
        # use kernel model of size (seq_length, 1) to get the global feature
        # x = F.max_pool2d(x, (x.size(2), 1)
        # x = x.squeeze(2)
        #x = F.max_pool2d(x,(x.size(1),1)).squeeze(2)

        # Time pool
        
        x = F.avg_pool2d(x,(x.size(2),1)).squeeze(2)
        # Vertexs Pool
        # x = F.avg_pool2d(x,(x.size(2),1)).squeeze(2)
        
        # Time x Vertex
        # x = torch.flatten(x, start_dim=1, end_dim=2)

        x = x.reshape(-1,1024)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_embedding = self.embedding(x.to(device))
        output_embedding = output_embedding.reshape(B,-1,768)
        return output_embedding.contiguous()