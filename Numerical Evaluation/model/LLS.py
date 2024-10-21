'''
@Filename       : LLS.py
@Description    : 
@Create_Time    : 2024/07/12 14:48:58
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import torch 
from torch import nn
from torch.nn.utils import spectral_norm as SN

def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega

class LLS(nn.Module):
    """
    Lifted Linear System(LLS) class
    """
    def __init__(self,lifting_dim,input_dim,output_dim):
        super(LLS, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encode_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, lifting_dim))
        
        self.decode_net = nn.Sequential(
            nn.Linear(lifting_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim))
        
        # koopman_dim = lifting_dim + state_dim
        koopman_dim = lifting_dim

        self.matrix_A = nn.Linear(koopman_dim, koopman_dim, bias = False)
        self.matrix_A.weight.data = gaussian_init_(koopman_dim, std=1)   #initial the weight of matrix_A 

    
    def encoder(self,x):
        """encoding state of system

        Args:
            z (tensor): state

        Returns:
            tensor: state after encoding
        """
        # z = torch.cat([x,self.encode_net(x)],axis=-1)
        return self.encode_net(x)
    
    def decoder(self,z):

        return self.decode_net(z)

    def forward(self, x):
        """compute the next step state of lifting linear system

        Args:
            z (tensor): state of lifting linear system
            u (tensor): control input of lifting linear system

        Returns:
            tensor: next step state of lifting linear system
        """
        z = self.matrix_A(self.encoder(x)) #compute the next step
        return self.decoder(z)
    
class LLS_wrapper(nn.Module):
    """
    Lifted Linear System(LLS) class
    """
    def __init__(self,state_dim,embedding_func,A,B):
        super(LLS_wrapper, self).__init__()

        self.embedding_func = embedding_func

        self.state_dim = state_dim

        self.A = A
        self.B = B

    def forward(self, x, u):
        """compute the next step state of lifting linear system

        Args:
            z (tensor): state of lifting linear system
            u (tensor): control input of lifting linear system

        Returns:
            tensor: next step state of lifting linear system
        """
        z = torch.matmul(self.A, self.embedding_func(x).t()) + torch.matmul(self.B, u.t())
        return z[:self.state_dim]
    

class LLS_no_decoder(nn.Module):
    """
    Lifted Linear System(LLS) class
    """
    def __init__(self,lifting_dim,input_dim):
        super(LLS_no_decoder, self).__init__()

        self.input_dim = input_dim

        # self.encode_net = nn.Sequential(
        #     nn.Linear(input_dim, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, lifting_dim))

        # self.encode_net = nn.Sequential(
        #     nn.Linear(input_dim, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, lifting_dim),
        #     nn.ReLU(True)
        #     )

        self.encode_net = nn.Sequential(
            SN(nn.Linear(input_dim, 128)),
            nn.ReLU(True),
            SN(nn.Linear(128, 128)),
            nn.ReLU(True),
            SN(nn.Linear(128, lifting_dim)),
            nn.ReLU(True)
            )
        
        # self.linear1 = SN(nn.Linear(input_dim, 128))
        # self.linear2 = SN(nn.Linear(128, 128))
        # self.linear3 = SN(nn.Linear(128,lifting_dim))
        
        # koopman_dim = lifting_dim + input_dim

        # self.matrix_A = nn.Linear(koopman_dim, koopman_dim, bias = False)
        # self.matrix_A.weight.data = gaussian_init_(koopman_dim, std=1)   #initial the weight of matrix_A 

    def forward(self,x):

        # z1 = self.linear1(x)
        # z2 = nn.ReLU(z1)
        # z3 = self.linear2(z2)
        # z4 = nn.ReLU(z3)
        # z5 = self.linear3(z4)
        # z6 = nn.ReLU(z5)

        z = torch.cat([x,self.encode_net(x)],axis=-1)

        return z

    # def encoder(self,x):
    #     """encoding state of system

    #     Args:
    #         z (tensor): state

    #     Returns:
    #         tensor: state after encoding
    #     """
    #     z = torch.cat([x,self.encode_net(x)],axis=-1)
    #     return z