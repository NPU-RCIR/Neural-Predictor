'''
@Filename       : BEM_Comparison.py
@Description    : 
@Create_Time    : 2024/07/24 11:13:08
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import os
import random
from datetime import datetime
import time
from copy import copy

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({
    "text.usetex": True
})

font = fm.FontProperties(family='Times New Roman',size=12, stretch=0)
fontdict = {'family': 'Times New Roman',
            'size': 15
            }
plt.rc('font',family='Times New Roman') 
lw = 1.5
fontsize = 15

current_folder = os.getcwd()
data_folder = os.path.join(current_folder,'data')


f_xy_NP = np.load(os.path.join(data_folder,'f_xy_NP.npy'))
f_z_NP = np.load(os.path.join(data_folder,'f_z_NP.npy'))
t_xy_NP = np.load(os.path.join(data_folder,'t_xy_NP.npy'))
t_z_NP = np.load(os.path.join(data_folder,'t_z_NP.npy'))


Bemnn_fxy_l = np.load(os.path.join(data_folder,'Bemnn_fxy_l.npy'))
Bemnn_fz_l = np.load(os.path.join(data_folder,'Bemnn_fz_l.npy'))
Bemnn_txy_l = np.load(os.path.join(data_folder,'Bemnn_txy_l.npy'))
Bemnn_tz_l = np.load(os.path.join(data_folder,'Bemnn_tz_l.npy'))


fxy_mhe_l = np.load(os.path.join(data_folder,'fxy_mhe_l.npy'))
fz_mhe_l = np.load(os.path.join(data_folder,'fz_mhe_l.npy'))
txy_mhe_l = np.load(os.path.join(data_folder,'txy_mhe_l.npy'))
tz_mhe_l = np.load(os.path.join(data_folder,'tz_mhe_l.npy'))


Gt_fxy_l = np.load(os.path.join(data_folder,'Gt_fxy_l.npy'))
Gt_fz_l = np.load(os.path.join(data_folder,'Gt_fz_l.npy'))
Gt_txy_l = np.load(os.path.join(data_folder,'Gt_txy_l.npy'))
Gt_tz_l = np.load(os.path.join(data_folder,'Gt_tz_l.npy'))

steps = len(f_xy_NP)
timesequence = np.linspace(0,1/400*steps,steps)

start = 45*400
end = 55*400
fig,ax = plt.subplots(2,2,figsize=(10,5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)


ax[0,0].plot(timesequence[start:end],fxy_mhe_l[start:end],label='NeuroMHE',color=plt.cm.Paired(8))
ax[0,0].plot(timesequence[start:end],Bemnn_fxy_l[start:end],label='NeuroBEM',color=plt.cm.Paired(4))
ax[0,0].plot(timesequence[start:end],f_xy_NP[start:end],label='NP',color=plt.cm.Paired(1))
ax[0,0].plot(timesequence[start:end],Gt_fxy_l[start:end],'--',label='Ground Truth',color=plt.cm.Paired(5))
# ax[0,0].legend()
ax[0,0].grid(True)
ax[0,0].set_xlim(timesequence[start]-0.5,timesequence[end]+0.5)
ax[0,0].set_xticks(np.arange(45, 56, 2))
# ax[0,0].set_xlabel(r't (s)')
ax[0,0].set_ylabel(r'$F_{xy}$ [N]',fontsize=fontsize)
axins = inset_axes(ax[0,0], width="35%", height="30%", loc='lower left',
                    bbox_to_anchor=(0.54, 0.60, 1, 1),
                    bbox_transform=ax[0,0].transAxes)
axins.plot(timesequence[start:end],fxy_mhe_l[start:end],label='NeuroMHE',color=plt.cm.Paired(8))
axins.plot(timesequence[start:end],Bemnn_fxy_l[start:end],label='NeuroBEM',color=plt.cm.Paired(4))
axins.plot(timesequence[start:end],f_xy_NP[start:end],label='NP',color=plt.cm.Paired(1))
axins.plot(timesequence[start:end],Gt_fxy_l[start:end],'--',label='Ground Truth',color=plt.cm.Paired(5))
axins.set_xlim(52.9,53.2)
axins.set_ylim(0.5,2.5)
axins.set_xticks([])
axins.set_yticks([])
mark_inset(ax[0,0], axins, loc1=4, loc2=3, fc="none", ec='k', lw=1)

ax[0,1].plot(timesequence[start:end],fz_mhe_l[start:end],label='NeuroMHE',color=plt.cm.Paired(8))
ax[0,1].plot(timesequence[start:end],Bemnn_fz_l[start:end],label='NeuroBEM',color=plt.cm.Paired(4))
ax[0,1].plot(timesequence[start:end],f_z_NP[start:end]+9.8*0.772,label='NP',color=plt.cm.Paired(1))
ax[0,1].plot(timesequence[start:end],Gt_fz_l[start:end],'--',label='Ground Truth',color=plt.cm.Paired(5))
# ax[0,0].legend()
ax[0,1].grid(True)
# ax[0,0].set_xlim(timesequence[start],timesequence[end])
ax[0,1].set_xlim(timesequence[start]-0.5,timesequence[end]+0.5)
ax[0,1].set_xticks(np.arange(45, 56, 2))
# ax[0,1].set_xlabel(r't (s)')
ax[0,1].set_ylabel(r'$F_{z}$ [N]',fontsize=fontsize)
axins = inset_axes(ax[0,1], width="20%", height="30%", loc='lower left',
                    bbox_to_anchor=(0.58, 0.60, 1, 1),
                    bbox_transform=ax[0,1].transAxes)
axins.plot(timesequence[start:end],fz_mhe_l[start:end],label='NeuroMHE',color=plt.cm.Paired(8))
axins.plot(timesequence[start:end],Bemnn_fz_l[start:end],label='NeuroBEM',color=plt.cm.Paired(4))
axins.plot(timesequence[start:end],f_z_NP[start:end]+9.8*0.772,label='NP',color=plt.cm.Paired(1))
axins.plot(timesequence[start:end],Gt_fz_l[start:end],'--',label='Ground Truth',color=plt.cm.Paired(5))
axins.set_xlim(50.1,50.4)
axins.set_ylim(16,22.5)
axins.set_xticks([])
axins.set_yticks([])
mark_inset(ax[0,1], axins, loc1=4, loc2=2, fc="none", ec='k', lw=1)

ax[1,0].plot(timesequence[start:end],txy_mhe_l[start:end],label='NeuroMHE',color=plt.cm.Paired(8))
ax[1,0].plot(timesequence[start:end],Bemnn_txy_l[start:end],label='NeuroBEM',color=plt.cm.Paired(4))
ax[1,0].plot(timesequence[start:end],t_xy_NP[start:end],label='NP',color=plt.cm.Paired(1))
ax[1,0].plot(timesequence[start:end],Gt_txy_l[start:end],'--',label='Ground Truth',color=plt.cm.Paired(5))

ax[1,0].grid(True)
# ax[0,0].set_xlim(timesequence[start],timesequence[end])
ax[1,0].set_xlim(timesequence[start]-0.5,timesequence[end]+0.5)
ax[1,0].set_xticks(np.arange(45, 56, 2))
ax[1,0].set_xlabel(r't [s]',fontsize=fontsize)
ax[1,0].set_ylabel(r'$\tau_{xy}$ [Nm]',fontsize=fontsize)
ax[1,0].legend(loc='upper right')
axins = inset_axes(ax[1,0], width="20%", height="40%", loc='lower left',
                    bbox_to_anchor=(0.10, 0.50, 1, 1),
                    bbox_transform=ax[1,0].transAxes)
axins.plot(timesequence[start:end],txy_mhe_l[start:end],label='NeuroMHE',color=plt.cm.Paired(8))
axins.plot(timesequence[start:end],Bemnn_txy_l[start:end],label='NeuroBEM',color=plt.cm.Paired(4))
axins.plot(timesequence[start:end],t_xy_NP[start:end],label='NP',color=plt.cm.Paired(1))
axins.plot(timesequence[start:end],Gt_txy_l[start:end],'--',label='Ground Truth',color=plt.cm.Paired(5))
axins.set_xlim(49.2,49.6)
axins.set_ylim(0.4,0.65)
axins.set_xticks([])
axins.set_yticks([])
mark_inset(ax[1,0], axins, loc1=4, loc2=2, fc="none", ec='k', lw=1)

ax[1,1].plot(timesequence[start:end],tz_mhe_l[start:end],label='NeuroMHE',color=plt.cm.Paired(8))
ax[1,1].plot(timesequence[start:end],Bemnn_tz_l[start:end],label='NeuroBEM',color=plt.cm.Paired(4))
ax[1,1].plot(timesequence[start:end],t_z_NP[start:end],label='NP',color=plt.cm.Paired(1))
ax[1,1].plot(timesequence[start:end],Gt_tz_l[start:end],'--',label='Ground Truth',color=plt.cm.Paired(5))
# ax[0,0].legend()
ax[1,1].grid(True)
# ax[0,0].set_xlim(timesequence[start],timesequence[end])
ax[1,1].set_xlim(timesequence[start]-0.5,timesequence[end]+0.5)
ax[1,1].set_xticks(np.arange(45, 56, 2))
ax[1,1].set_xlabel(r't [s]',fontsize=fontsize)
ax[1,1].set_ylabel(r'$\tau_{z}$ [Nm]',fontsize=fontsize)
# ax[1,1].legend()
axins = inset_axes(ax[1,1], width="30%", height="25%", loc='lower left',
                    bbox_to_anchor=(0.58, 0.63, 1, 1),
                    bbox_transform=ax[1,1].transAxes)
axins.plot(timesequence[start:end],tz_mhe_l[start:end],label='NeuroMHE',color=plt.cm.Paired(8))
axins.plot(timesequence[start:end],Bemnn_tz_l[start:end],label='NeuroBEM',color=plt.cm.Paired(4))
axins.plot(timesequence[start:end],t_z_NP[start:end],label='NP',color=plt.cm.Paired(1))
axins.plot(timesequence[start:end],Gt_tz_l[start:end],'--',label='Ground Truth',color=plt.cm.Paired(5))
axins.set_xlim(50.75,51.25)
axins.set_ylim(-0.02,0.02)
axins.set_xticks([])
axins.set_yticks([])
mark_inset(ax[1,1], axins, loc1=4, loc2=2, fc="none", ec='k', lw=1)

plt.savefig('bem_comparison.png',dpi=600,transparent=True,bbox_inches ='tight')

# plt.show()