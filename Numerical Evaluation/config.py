'''
@Filename       : config.py
@Description    : 
@Create_Time    : 2024/07/12 21:32:55
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import argparse
import numpy as np

parser = argparse.ArgumentParser()

'''general'''
parser.add_argument('--epoch', type=int, default=800, help='epoch')
parser.add_argument('--bs', type=int, default=32, help='batch size')

'''parameters of dataset'''
# parser.add_argument('--steps', type=int, default=200, help='steps of trajectory of dataset')
# parser.add_argument('--n_trj', type=int, default=20, help='number of trajectory of dataset')
# parser.add_argument('--predict_time', type=int, default=20, help='')

'''configuration of tensorboard writer'''
parser.add_argument('--experiment', type=str, default='LLS', help='')
parser.add_argument('--model_name', type=str, default='', help='')
parser.add_argument('--extra', type=str, default='', help='')

parser.add_argument('--lifting_dim', type=int, default=20, help='')


def gen_args():
    args = parser.parse_args()

    return args