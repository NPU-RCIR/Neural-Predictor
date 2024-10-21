import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import linalg as LA

current_folder = os.getcwd()
process_data_folder = os.path.join(current_folder,'processed_data')

sys_param  = np.array([0.752,0.0025,0.0021,0.0043]) 
m = sys_param[0]
J = np.diag([sys_param[1],sys_param[2],sys_param[3]])

"""---------------------------------Quaternion to Rotation Matrix---------------------------"""
def Quaternion2Rotation(q): 
    q = q/LA.norm(q) # normalization, which is very important to guarantee that the resulting R is a rotation matrix (Lie group: SO3)
    q0, q1, q2, q3 = q[0,0], q[1,0], q[2,0], q[3,0]
    R = np.array([
        [2 * (q0 ** 2 + q1 ** 2) - 1, 2 * q1 * q2 - 2 * q0 * q3, 2 * q0 * q2 + 2 * q1 * q3],
        [2 * q0 * q3 + 2 * q1 * q2, 2 * (q0 ** 2 + q2 ** 2) - 1, 2 * q2 * q3 - 2 * q0 * q1],
        [2 * q1 * q3 - 2 * q0 * q2, 2 * q0 * q1 + 2 * q2 * q3, 2 * (q0 ** 2 + q3 ** 2) - 1]
    ]) # from body frame to inertial frame
    return R

"""---------------------------------Compute Ground Truth------------------------------------"""
def GroundTruth(w_B, acc, mass, J_B):
    acc_p = np.array([[acc[0, 0], acc[1, 0], acc[2, 0]]]).T # measured in body frame, already including the gravity
    acc_w = np.array([[acc[3, 0], acc[4, 0], acc[5, 0]]]).T # measured in body frame
    
    df    = mass*acc_p
    dt    = np.matmul(J_B, acc_w) + \
            np.cross(w_B.T, np.transpose(np.matmul(J_B, w_B))).T 
    return df, dt


train_set = {'a': "merged_2021-02-23-14-41-07_seg_3",
             'b': "merged_2021-02-03-13-44-49_seg_3"}

evaluate_set = {'a':"merged_2021-02-18-13-44-23_seg_2",
                'b':"merged_2021-02-18-16-53-35_seg_2",
                'c':"merged_2021-02-18-17-03-20_seg_2",
                'd':"merged_2021-02-18-17-19-08_seg_2",
                'e':"merged_2021-02-18-17-26-00_seg_1",
                'f':"merged_2021-02-18-18-08-45_seg_1",
                'g':"merged_2021-02-23-10-48-03_seg_2",
                'h':"merged_2021-02-23-11-41-38_seg_3",
                'i':"merged_2021-02-23-14-21-48_seg_3",
                'j':"merged_2021-02-23-17-27-24_seg_2",
                'k':"merged_2021-02-23-19-45-06_seg_2",
                'l':"merged_2021-02-23-22-26-25_seg_2",
                'm':"merged_2021-02-23-22-54-17_seg_1"} 

def process_csv_data(csv_name:str, 
                     options='train', 
                     plot_save=False):

    dataset = os.path.join(process_data_folder,csv_name+'.csv') # the acc data includes the gravity acc

    dataset = pd.read_csv(dataset)
    dataframe = pd.DataFrame(dataset)

    angaccx_seq, angaccy_seq, angaccz_seq = dataframe['ang acc x'], dataframe['ang acc y'], dataframe['ang acc z']
    angvelx_seq, angvely_seq, angvelz_seq = dataframe['ang vel x'], dataframe['ang vel y'], dataframe['ang vel z']
    qx_seq, qy_seq, qz_seq, qw_seq = dataframe['quat x'], dataframe['quat y'], dataframe['quat z'], dataframe['quat w']
    accx_seq, accy_seq, accz_seq = dataframe['acc x'], dataframe['acc y'], dataframe['acc z']
    velx_seq, vely_seq, velz_seq = dataframe['vel x'], dataframe['vel y'], dataframe['vel z']
    posx_seq, posy_seq, posz_seq = dataframe['pos x'], dataframe['pos y'], dataframe['pos z']
    moto1_seq, moto2_seq, moto3_seq, moto4_seq = dataframe['mot 1'], dataframe['mot 2'], dataframe['mot 3'], dataframe['mot 4']

    if options == 'train':
        n_start = 500
        data_point_num = 4000
    if options == 'test':
        n_start = 0
        data_point_num = len(accx_seq)

    dataset_processed = np.zeros((12,data_point_num))

    for index in range(data_point_num):
        '''calculate external force'''
        v_B        = np.array([[velx_seq[n_start+index], vely_seq[n_start+index], velz_seq[n_start+index]]]).T
        q          = np.array([[qw_seq[n_start+index], qx_seq[n_start+index], qy_seq[n_start+index], qz_seq[n_start+index]]]).T
        w_B        = np.array([[angvelx_seq[n_start+index], angvely_seq[n_start+index], angvelz_seq[n_start+index]]]).T
        R_B        = Quaternion2Rotation(q)  
        v_I        = np.matmul(R_B, v_B)
        acc        = np.array([[accx_seq[n_start+index], accy_seq[n_start+index], accz_seq[n_start+index]-9.8, angaccx_seq[n_start+index], angaccy_seq[n_start+index], angaccz_seq[n_start+index]]]).T
        df_t, dt_t = GroundTruth(w_B, acc, m, J)
        # df_t       = np.matmul(R_B0, df_t) # transformed to world frame
        df_B       = np.reshape(df_t, (3, 1))
        dt_B       = np.reshape(dt_t, (3, 1))
        moto       = np.array([[moto1_seq[n_start+index], moto2_seq[n_start+index], moto3_seq[n_start+index], moto4_seq[n_start+index]]]).T
        
        dataset_processed[:,index] = np.squeeze(np.concatenate((v_B,w_B,df_B,dt_B),axis=0))

    ''' plot trj '''
    if plot_save:
        fig,ax = plt.subplots(figsize=(5,5))
        ax.plot(posx_seq, posy_seq)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('off')
        plt.savefig(os.path.join(csv_name+'_trj'),dpi=600,transparent=True,bbox_inches ='tight')

    np.save(os.path.join(csv_name),dataset_processed.T)

options = 'train'

if options == 'test':
    for csv_name in evaluate_set.values():
        process_csv_data(csv_name, options='test',plot_save=True)
if options == 'train':
    for csv_name in train_set.values():
        process_csv_data(csv_name, options='train',plot_save=True)