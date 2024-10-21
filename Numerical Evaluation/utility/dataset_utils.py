'''
@Filename       : data_utils.py
@Description    : 
@Create_Time    : 2024/07/17 20:05:39
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

import numpy as np

def process_data(data,segment_size):
    """function for processing data

    Args:
        data (array)

    Returns:
        dataset:  processed dataset [_, segment_size, n]
    """
    data_length, n = data.shape

    dataset = np.zeros((int(data_length/segment_size), segment_size, n))

    for i in range(int(data_length/segment_size)):
        dataset[i,:,:] = data[i*segment_size:(i+1)*segment_size]

    return dataset