'''
@Filename       : validation_MHE.py
@Description    : 
@Create_Time    : 2024/07/17 21:27:09
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import root_mean_squared_error
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm

current_folder = os.getcwd()
log_folder_prefix = os.path.join(current_folder,'dump')

device = "cuda" if torch.cuda.is_available() else "cpu"

''' BEM validation '''

data_folder = os.path.join(current_folder,'data','BEM')

train_data_prefix = ['merged_2021-02-23-14-41-07_seg_3']
test_data_prefix = ["merged_2021-02-18-13-44-23_seg_2",
                    "merged_2021-02-18-16-53-35_seg_2",
                    "merged_2021-02-18-17-03-20_seg_2",
                    "merged_2021-02-18-17-19-08_seg_2",
                    "merged_2021-02-18-17-26-00_seg_1",
                    "merged_2021-02-18-18-08-45_seg_1",
                    "merged_2021-02-23-10-48-03_seg_2",
                    "merged_2021-02-23-11-41-38_seg_3",
                    "merged_2021-02-23-14-21-48_seg_3",
                    "merged_2021-02-23-17-27-24_seg_2",
                    "merged_2021-02-23-19-45-06_seg_2",
                    "merged_2021-02-23-22-26-25_seg_2",
                    "merged_2021-02-23-22-54-17_seg_1" ]

trj_name = ['3D Circle_1','Linear Oscillation', 'Figure-8_1', 'Race Track_1', 'Race Track_2', '3D Circle_2', 'Figure-8_2', 'Melon_1', 
       'Figure-8_3', 'Figure-8_4','Melon_2','Random Points', 'Ellipse']

NeuroBEM = np.array([[0.196, 0.211, 0.215, 0.005, 0.006, 0.003, 0.288, 0.008, 0.360, 0.009],
                     [0.164, 0.185, 0.456, 0.013, 0.011, 0.006, 0.247, 0.017, 0.518, 0.018],
                     [0.065, 0.056, 0.235, 0.004, 0.003, 0.002, 0.085, 0.005, 0.250, 0.006],
                     [0.169, 0.158, 0.463, 0.009, 0.009, 0.004, 0.231, 0.013, 0.517, 0.013],
                     [0.262, 0.248, 0.552, 0.014, 0.012, 0.007, 0.360, 0.019, 0.659, 0.020],
                     [0.110, 0.129, 0.470, 0.006, 0.009, 0.004, 0.170, 0.011, 0.499, 0.011],
                     [0.051, 0.036, 0.339, 0.002, 0.002, 0.002, 0.063, 0.003, 0.345, 0.003],
                     [0.099, 0.108, 0.397, 0.004, 0.005, 0.003, 0.147, 0.007, 0.423, 0.007],
                     [0.145, 0.168, 0.584, 0.010, 0.012, 0.006, 0.221, 0.015, 0.624, 0.017],
                     [0.400, 0.313, 1.084, 0.020, 0.018, 0.009, 0.508, 0.027, 1.197, 0.028],
                     [0.244, 0.198, 0.921, 0.009, 0.006, 0.003, 0.314, 0.015, 0.974, 0.016],
                     [0.161, 0.183, 0.471, 0.008, 0.008, 0.005, 0.244, 0.012, 0.530, 0.013],
                     [0.204, 0.315, 1.039, 0.012, 0.008, 0.005, 0.375, 0.022, 1.105, 0.023]])

NeuroMHE = np.array([[0.258, 0.269, 0.108, 0.003, 0.002, 0.003, 0.373, 0.004, 0.388, 0.005],
                     [0.119, 0.105, 0.186, 0.011, 0.007, 0.005, 0.159, 0.013, 0.244, 0.014],
                     [0.039, 0.056, 0.039, 0.002, 0.001, 0.002, 0.069, 0.002, 0.079, 0.003],
                     [0.141, 0.092, 0.115, 0.007, 0.004, 0.004, 0.168, 0.009, 0.204, 0.009],
                     [0.245, 0.175, 0.208, 0.012, 0.008, 0.018, 0.301, 0.014, 0.366, 0.023],
                     [0.140, 0.135, 0.075, 0.003, 0.002, 0.004, 0.194, 0.004, 0.208, 0.006],
                     [0.020, 0.058, 0.029, 0.002, 0.001, 0.002, 0.061, 0.002, 0.068, 0.003],
                     [0.053, 0.059, 0.060, 0.003, 0.001, 0.002, 0.079, 0.003, 0.099, 0.004],
                     [0.118, 0.133, 0.151, 0.010, 0.006, 0.005, 0.178, 0.012, 0.233, 0.013],
                     [0.169, 0.174, 0.237, 0.010, 0.010, 0.007, 0.242, 0.017, 0.339, 0.018],
                     [0.254, 0.213, 0.094, 0.005, 0.003, 0.004, 0.331, 0.005, 0.344, 0.007],
                     [0.115, 0.114, 0.204, 0.010, 0.006, 0.005, 0.162, 0.012, 0.260, 0.012],
                     [0.176, 0.165, 0.089, 0.005, 0.003, 0.006, 0.242, 0.006, 0.258, 0.008]])

trj_num = len(trj_name)

method_comparison = ['NeuroBEM', 'NeuroMHE', 'Neuro_Predictor']

validate_option = 'test'
data_list = []

if validate_option == 'test': data_list = test_data_prefix
if validate_option == 'train': data_list = train_data_prefix

state_dim = 6
u_dim = 6

rmse = {'fx':[],
        'fy':[],
        'fz':[],
        'tx':[],
        'ty':[],
        'tz':[],
        'fxy':[],
        'txy':[],
        'f':[],
        't':[]}


experiment = 'evaluation'
    
experiment_folder = os.path.join(log_folder_prefix,experiment)

res_NP = np.zeros((trj_num,10))

for num, each_data in enumerate(data_list):
    dataset = np.load(os.path.join(data_folder,each_data+'.npy'))
    steps = dataset.shape[0]
    select_trj = dataset
    pre_trj = np.zeros((steps,6))
    timesequence = np.linspace(0,1/400*steps,steps)

    pth_model = os.path.join(experiment_folder, 'lifting_func.pth')
    net = torch.load(pth_model,map_location='cpu')
    net.eval()

    matrix_path = os.path.join(experiment_folder)
    A = torch.load(os.path.join(matrix_path,'A.pth'),map_location='cpu')
    B = torch.load(os.path.join(matrix_path,'B.pth'),map_location='cpu')

    lls_path = os.path.join(experiment_folder, 'lls_wrapper.pth')
    lls = torch.load(lls_path,map_location='cpu')

    pre_trj[0,:] = select_trj[0,u_dim:]
    
    for i in range(steps-1):
        # start = time.perf_counter()
        x = select_trj[i,u_dim:]
        u = select_trj[i,:u_dim]
        # z = net(torch.from_numpy(x).reshape(1,state_dim))

        # z_next = torch.matmul(A, z.t()) + torch.matmul(B, torch.from_numpy(u).reshape(1,u_dim).t())

        z_next = lls(torch.from_numpy(x).reshape(1,state_dim),torch.from_numpy(u).reshape(1,u_dim))

        pre_trj[i+1,:] = z_next[:state_dim,:].squeeze(1).detach().numpy()
        # end = time.perf_counter()
        # print("runtime", end-start)
    
    dim  = u_dim

    RMSE_fx = root_mean_squared_error(select_trj[:,dim],pre_trj[:,0])
    RMSE_fy = root_mean_squared_error(select_trj[:,dim+1],pre_trj[:,1])
    RMSE_fz = root_mean_squared_error(select_trj[:,dim+2],pre_trj[:,2])
    RMSE_tx = root_mean_squared_error(select_trj[:,dim+3],pre_trj[:,3])
    RMSE_ty = root_mean_squared_error(select_trj[:,dim+4],pre_trj[:,4])
    RMSE_tz = root_mean_squared_error(select_trj[:,dim+5],pre_trj[:,5])

    e_fxy = []
    e_f = []
    e_txy = []
    e_t = []
    zeros_vector_ = []

    f_xy_true = []
    f_z_true  = []
    t_xy_true =[]
    t_z_true  = []

    f_xy_NP = []
    f_z_NP  = []
    t_xy_NP =[]
    t_z_NP  = []

    for index in range(steps):
        e_fx = (select_trj[index,dim]-pre_trj[index,0])
        e_fy = (select_trj[index,dim+1]-pre_trj[index,1])
        e_fz = (select_trj[index,dim+2]-pre_trj[index,2])
        e_fxy.append(np.sqrt(e_fx**2+e_fy**2))
        e_f.append(np.sqrt(e_fx**2+e_fy**2+e_fz**2))

        e_tx = (select_trj[index,dim+3]-pre_trj[index,3])
        e_ty = (select_trj[index,dim+4]-pre_trj[index,4])
        e_tz = (select_trj[index,dim+5]-pre_trj[index,5])
        e_txy.append(np.sqrt(e_tx**2+e_ty**2))
        e_t.append(np.sqrt(e_tx**2+e_ty**2+e_tz**2))

        zeros_vector_.append(0)

        f_xy_true.append(np.sqrt((select_trj[index,dim]**2+select_trj[index,dim+1]**2)))
        f_z_true.append(select_trj[index,dim+2])
        t_xy_true.append(np.sqrt(select_trj[index,dim+3]**2+select_trj[index,dim+4]**2))
        t_z_true.append(select_trj[index,dim+5])

        f_xy_NP.append(np.sqrt((pre_trj[index,0]**2+pre_trj[index,1]**2)))
        f_z_NP.append(pre_trj[index,2])
        t_xy_NP.append(np.sqrt(pre_trj[index,3]**2+pre_trj[index,4]**2))
        t_z_NP.append(pre_trj[index,5])

    RMSE_fxy = root_mean_squared_error(e_fxy,zeros_vector_)
    RMSE_f = root_mean_squared_error(e_f,zeros_vector_)
    RMSE_txy = root_mean_squared_error(e_txy,zeros_vector_)
    RMSE_t = root_mean_squared_error(e_t,zeros_vector_)

    res_NP[num,:] = np.array([RMSE_fx,RMSE_fy,RMSE_fz,RMSE_tx,RMSE_ty,RMSE_tz,RMSE_fxy,RMSE_txy,RMSE_f,RMSE_t])
    
    fig,ax = plt.subplots(3,1,figsize=(10,8))
    ax[0].plot(timesequence,select_trj[:,dim],color=plt.cm.tab20c(2))
    ax[0].plot(timesequence,pre_trj[:,0],color='k')
    # ax[0].set_xlabel('t/s')
    ax[0].set_ylabel('F_x/N')
    ax[1].plot(timesequence,select_trj[:,dim+1],color=plt.cm.tab20c(2))
    ax[1].plot(timesequence,pre_trj[:,1],color='k')
    # ax[1].set_xlabel('t/s')
    ax[1].set_ylabel('F_y/N')
    ax[2].plot(timesequence,select_trj[:,dim+2],color=plt.cm.tab20c(2),label='true')
    ax[2].plot(timesequence,pre_trj[:,2],color='k',label='predicted')
    ax[2].set_xlabel('t/s')
    ax[2].set_ylabel('F_z/N')
    ax[2].legend()

    fig_path = os.path.join(experiment_folder, validate_option, each_data)
    if not os.path.exists(fig_path): os.makedirs(fig_path)
    plt.savefig(os.path.join(fig_path, '_force.png'),dpi=600,transparent=True,bbox_inches ='tight')
    plt.close()

    fig,ax_ = plt.subplots(3,1,figsize=(10,8))
    ax_[0].plot(timesequence,select_trj[:,dim+3],color=plt.cm.tab20c(2))
    ax_[0].plot(timesequence,pre_trj[:,3],color='k')
    # ax_[0].set_xlabel('t/s')
    ax_[0].set_ylabel('T_x/(N.m)')
    ax_[1].plot(timesequence,select_trj[:,dim+4],color=plt.cm.tab20c(2))
    ax_[1].plot(timesequence,pre_trj[:,4],color='k')
    # ax_[1].set_xlabel('t/s')
    ax_[1].set_ylabel('T_y/(N.m)')
    ax_[2].plot(timesequence,select_trj[:,dim+5],color=plt.cm.tab20c(2),label='true')
    ax_[2].plot(timesequence,pre_trj[:,5],color='k',label='predicted')
    ax_[2].set_xlabel('t/s')
    ax_[2].set_ylabel('T_z/(N.m)')
    ax_[2].legend()
    plt.savefig(os.path.join(fig_path, '_torque.png'),dpi=600,transparent=True,bbox_inches ='tight')
    plt.close()

    with open(os.path.join(experiment_folder, validate_option, each_data, 'rmse_results.txt'),'w') as file:
            file.write('RMSE_fx: '+ str(RMSE_fx) + '\r\n')
            file.write('RMSE_fy: '+ str(RMSE_fy) + '\r\n')
            file.write('RMSE_fz: '+ str(RMSE_fz) + '\r\n')
            file.write('RMSE_tx: '+ str(RMSE_tx) + '\r\n')
            file.write('RMSE_ty: '+ str(RMSE_ty) + '\r\n')
            file.write('RMSE_tz: '+ str(RMSE_tz) + '\r\n')
            file.write('RMSE_fxy: '+ str(RMSE_fxy) + '\r\n')
            file.write('RMSE_txy: '+ str(RMSE_txy) + '\r\n')
            file.write('RMSE_f: '+ str(RMSE_f) + '\r\n')
            file.write('RMSE_t: '+ str(RMSE_t) + '\r\n')


    rmse['fx'].append(RMSE_fx)
    rmse['fy'].append(RMSE_fy)
    rmse['fz'].append(RMSE_fz)
    rmse['fxy'].append(RMSE_fxy)
    rmse['f'].append(RMSE_f)
    rmse['tx'].append(RMSE_tx)
    rmse['ty'].append(RMSE_ty)
    rmse['tz'].append(RMSE_tz)
    rmse['txy'].append(RMSE_txy)
    rmse['t'].append(RMSE_t)

    np.save(os.path.join(experiment_folder, validate_option, each_data,'f_xy_true.npy'),f_xy_true)
    np.save(os.path.join(experiment_folder, validate_option, each_data,'f_z_true.npy'),f_z_true)
    np.save(os.path.join(experiment_folder, validate_option, each_data,'t_xy_true.npy'),t_xy_true)
    np.save(os.path.join(experiment_folder, validate_option, each_data,'t_z_true.npy'),t_z_true)
    np.save(os.path.join(experiment_folder, validate_option, each_data,'f_xy_NP.npy'),f_xy_NP)
    np.save(os.path.join(experiment_folder, validate_option, each_data,'f_z_NP.npy'),f_z_NP)
    np.save(os.path.join(experiment_folder, validate_option, each_data,'t_xy_NP.npy'),t_xy_NP)
    np.save(os.path.join(experiment_folder, validate_option, each_data,'t_z_NP.npy'),t_z_NP)



methods = []
trjs = []

res = np.zeros((trj_num*len(method_comparison),10)) # results of method comparison

res_NP = np.around(res_NP,3)

for each in range(trj_num):
    for i in range(len(method_comparison)):
        methods.append(method_comparison[i])
    trjs.append(trj_name[each])
    trjs.append('')
    trjs.append('')

    res[each*3,:] = NeuroBEM[each,:]
    res[each*3+1,:] = NeuroMHE[each,:]
    res[each*3+2,:] = res_NP[each,:]

data = pd.DataFrame({'trj_name':trjs,
                    'methods':methods,
                    'f_x':res[:,0],
                    'f_y':res[:,1],
                    'f_z':res[:,2],
                    't_x':res[:,3],
                    't_y':res[:,4],
                    't_z':res[:,5],
                    'f_xy':res[:,6],
                    't_xy':res[:,7],
                    'f':res[:,8],
                    't':res[:,9]})

data.to_csv(os.path.join(experiment_folder,'rmse_result'+'.csv'),mode='w',index=False,header=True,sep=',')