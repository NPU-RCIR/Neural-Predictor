'''
@Filename       : MHE_training.py
@Description    : 
@Create_Time    : 2024/07/17 20:03:57
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import os
import random

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from model.LLS import LLS_no_decoder, LLS_wrapper
from utility.training_utils import create_writer
from utility.dataset_utils import process_data
from config import gen_args

current_folder = os.getcwd()
data_folder = os.path.join(current_folder,'data')
log_folder = os.path.join(current_folder,'dump')
bem_folder = os.path.join(data_folder,'BEM')

# set random seed value for reproducing the results presented in the paper
seed_value = 723054704  
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  

torch.manual_seed(seed_value)     # random seed value for CPU
torch.cuda.manual_seed(seed_value)      # random seed value for GPU
torch.cuda.manual_seed_all(seed_value)   # random seed value for GPU (multi-GPUs)

torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"

''' BEM DATASET '''
train_set = {'a': "merged_2021-02-03-13-44-49_seg_3"}

'''
dataset = [segment nums, size for each segment, n]
'''

def train(args,
          epochs: int,
          predict_time:int,
          state_dim: int,
          u_dim: int,
          net:torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          loss_function: nn.Module,
          dataset,
          writer: torch.utils.tensorboard.writer.SummaryWriter,
          log_dir: str,
          ):

    segment_num, steps, n = dataset.shape

    assert n == state_dim + u_dim

    pbar = tqdm(total=epochs,desc='description')

    best_loss = 10

    for epoch in range(epochs):
        x = dataset[:,:-predict_time,u_dim:].reshape(segment_num*(steps-predict_time),state_dim)
        u = dataset[:,:-predict_time,:u_dim].reshape(segment_num*(steps-predict_time),u_dim)
        y = dataset[:,1:-(predict_time-1),u_dim:].reshape(segment_num*(steps-predict_time),state_dim)
        
        z = net((x))
        z_next = net((y))

        W = torch.cat((z.t(),u.t()),0)
        V = z_next.t()

        Vwt = torch.matmul(V,W.t())
        Wwt = torch.matmul(W,W.t())

        AB = torch.matmul(Vwt,torch.pinverse(Wwt))
        A = AB[:,0:-u_dim]
        B = AB[:,-u_dim:]

        x_ = dataset[:,predict_time:,u_dim:].reshape(segment_num*(steps-predict_time),state_dim)

        for step in range(predict_time-1):
            u = dataset[:,step:-(predict_time-step),:u_dim].reshape(segment_num*(steps-predict_time),u_dim)
            z_next = torch.matmul(A,z.t()) + torch.matmul(B,u.t())
            z = net(z_next[:state_dim,:].t())
        
        multi_step_prediction_loss = loss_function(x_,z[:,:state_dim])

        loss = multi_step_prediction_loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        pbar.update(1)
        pbar.set_description('loss: %.2e' % (multi_step_prediction_loss.cpu().detach().numpy()))
        writer.add_scalars(main_tag='train_loss',
                           tag_scalar_dict={'loss':multi_step_prediction_loss,
                                            'multi_step_predcition_loss':multi_step_prediction_loss},
                           global_step=epoch)
        
        if loss < best_loss:
            best_loss = loss
            torch.save(net,(os.path.join(log_dir,'lifting_func.pth')))
            torch.save(A,os.path.join(log_dir,'A.pth'))
            torch.save(B,os.path.join(log_dir,'B.pth'))
            
            lls_wrapper = LLS_wrapper(state_dim, net, A, B)
            torch.save(lls_wrapper,os.path.join(log_dir,'lls_wrapper.pth'))

    with open(os.path.join(log_dir,'log.txt'),'w') as log:
            log.write("args:" + str(args) + "\r\n" + "best_loss: {}".format(best_loss))


if __name__ == '__main__':

    args = gen_args()

    epochs = args.epoch
    batch_size = args.bs
    learning_rate = 1e-3

    lifting_dim = args.lifting_dim

    experiment_name = args.experiment
    model_name = args.model_name

    data = np.load(os.path.join(bem_folder,train_set['a']+'.npy'))

    dataset = process_data(data,40)

    dataset = torch.from_numpy(dataset).to(device)

    writer, log_dir = create_writer(folder=log_folder, 
                                    experiment_name=args.experiment, 
                                    model_name=args.model_name,
                                    extra=args.extra)
    
    ''' state_dim: dimension of force and torque, u_dim: dimension of 'control input' of LLS '''
    state_dim, u_dim = 6,6
    
    net = LLS_no_decoder(lifting_dim=lifting_dim, input_dim=state_dim).to(device)
    net.double()

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    loss_func = nn.MSELoss()

    train(args=args,
          epochs=epochs,
          predict_time=20,
          state_dim=state_dim,
          u_dim=u_dim,
          net=net,
          optimizer=optimizer,
          loss_function=loss_func,
          dataset=dataset,
          writer=writer,
          log_dir=log_dir)

    