'''
@Filename       : quadrotor.py
@Description    : 
@Create_Time    : 2024/07/12 09:41:09
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import torch
from torch import nn   
from model.LLS import LLS

class quadrotor_LLS(nn.Module):
    def __init__(self,device,lifting_dim):
        super(quadrotor_LLS, self).__init__()
        self.device = device
        self.state_dim = 12
        self.u_dim = 4
        self.delta_t = 0.025
        self.mass = torch.tensor([2]).to(device)
        self.hover_thrust = torch.tensor([0.483272]).to(device)
        self.g = torch.tensor([9.81]).to(device)

        self.external_force = LLS(lifting_dim=lifting_dim,input_dim=6,output_dim=6)
    
    def dynamics(self,t,x):
        bs, n = x.size()
        assert n == self.state_dim+self.u_dim
        
        dx = torch.zeros_like(x)

        p_x,p_y,p_z = x[:,0],x[:,1],x[:,2]
        vx,vy,vz = x[:,3],x[:,4],x[:,5]
        phi,theta,psi = x[:,6],x[:,7],x[:,8]
        p,q,r = x[:,9],x[:,10],x[:,11]

        a_thrust = x[:,12]

        # Time-derivative of the position vector
        dx[:,0] = vx
        dx[:,1] = vy
        dx[:,2] = vz

        # Time-derivative of the velocity vector
        dx[:,3] = (torch.sin(phi)*torch.sin(psi) + torch.sin(theta)*torch.cos(phi)*torch.cos(psi))*(a_thrust/self.hover_thrust)*self.g
        dx[:,4] = (torch.sin(phi)*torch.cos(psi) - torch.sin(psi)*torch.sin(theta)*torch.cos(phi))*(a_thrust/self.hover_thrust)*self.g
        dx[:,5] = torch.cos(phi)*torch.cos(theta)*(a_thrust/self.hover_thrust)*self.g # the acceleration in dataset not contains gravity

        # calculate external force
        external_force = self.external_force(x[:,:6])

        dx[:,:6] = dx[:,:6]+ external_force

        return dx
    
    def discrete_dynamics(self,t,x):
        """
        brief: using runge-kutta4 method for discreting quadrotor dynamics 
        param: t,x,u
        return: next step state
        """
        k1 = self.dynamics(t,x)
        k2 = self.dynamics(t,x + k1*self.delta_t/2)
        k3 = self.dynamics(t,x + k2*self.delta_t/2)
        k4 = self.dynamics(t,x + k1*self.delta_t)
        return (x + (self.delta_t/6)*(k1+2*k2+2*k3+k4))
        
    def forward(self,x):
        return self.discrete_dynamics(0,x)

class loss():
    def __init__(self,
                 net: quadrotor_LLS,
                 loss_function: torch.nn.Module,
                 device):
        self.net = net
        self.loss_function = loss_function
        self.device = device

    def forward_multi_step_loss(self,
                                dataset):
        
        bs, steps, n = dataset.size()
        assert n == self.net.state_dim + self.net.u_dim

        loss = torch.zeros(1,dtype=torch.float64).to(self.device)

        inital_state = dataset[:,0,:]
        for i in range(steps-1):
            z = self.net(inital_state)
            loss += self.loss_function(z[:,:6], dataset[:,i+1,:6])
            inital_state = torch.cat([z[:,:6],dataset[:,i+1,6:]],1)
        
        return loss/steps
        
    def stable_constraint_loss(self,
                               dataset):
        A = self.net.external_force.matrix_A.weight

        Ak  = A

        c = torch.linalg.eigvals(Ak).abs()-torch.ones(1,dtype=torch.float64).to(self.device)
        mask = c>0
        return c[mask].sum()
    
    
    def losses(self,
               dataset):

        losses = self.forward_multi_step_loss(dataset) + self.stable_constraint_loss(dataset)

        return losses