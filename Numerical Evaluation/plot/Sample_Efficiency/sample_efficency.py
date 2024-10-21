'''
@Filename       : sample_efficency.py
@Description    : 
@Create_Time    : 2024/07/30 17:37:35
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
fontsize = 10

''' rmse of race_track_1 '''
samples_1k_f = 0.357	
samples_1k_t = 0.094

samples_2k_f = 0.156	
samples_2k_t = 0.022

samples_3k_f = 0.127	
samples_3k_t = 0.011

samples_4k_f = 0.076	
samples_4k_t = 0.007

samples_5k_f = 0.078	
samples_5k_t = 0.007

samples_6k_f = 0.082
samples_6k_t = 0.006

NeuroMHE_f = 0.204	
NeuroMHE_t = 0.009

NeuroMHE_f = np.linspace(NeuroMHE_f,NeuroMHE_f,6)
NeuroMHE_t = np.linspace(NeuroMHE_t,NeuroMHE_t,6)

x = [1,2,3,4,5,6]
f = [samples_1k_f,samples_2k_f,samples_3k_f,samples_4k_f,samples_5k_f,samples_6k_f]
t = [samples_1k_t,samples_2k_t,samples_3k_t,samples_4k_t,samples_5k_t,samples_6k_t]

lw =2
fig,ax = plt.subplots(figsize=(6,3))

ax.plot(x,f,'k*--', alpha=0.5, linewidth=lw, label='NP_F')
ax.plot(x,NeuroMHE_f, 'k-.', alpha=0.5, linewidth=1,label='NeuroMHE_F')
ax.spines['right'].set_visible(False)
ax.set_xlabel('Samples')
ax.set_ylabel(r'F [N]')
ax.grid(True)
# ax.legend(loc='upper left')

t_ax = ax.twinx()
t_ax.plot(x,t,'bo--', alpha=0.5, linewidth=lw, label='NP_t')
t_ax.plot(x,NeuroMHE_t,'b-.',alpha=0.5, linewidth=1,label='NeuroMHE_t')
t_ax.set_ylim(0.005,0.11)
t_ax.set_yticks(np.arange(0,0.11,0.02))
t_ax.spines['right'].set(color='b', linewidth=2.0, linestyle=':')
t_ax.tick_params(length=6, width=2, color='b', labelcolor='b')
t_ax.set_ylabel(r'$\tau$ [Nm]')
# t_ax.legend(loc='upper right')

plt.savefig('sample_efficiency.png',dpi=600,transparent=True,bbox_inches ='tight')

plt.show()