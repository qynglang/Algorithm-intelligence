import numpy as np
#import matplotlib.pyplot as plt
from rotate import get_img_rot_broa as rotate
from loadone import load_sample as load
from convo import conv2d,maxpool2d,conv_net
import tensorflow as tf
from con import train_cov
def wei():
    data=np.load('data.npy')
    weight=np.zeros((5,5,1,3,8,36))
    weightt=np.zeros((5,5,3,6,8,36))
    for i in range (0,36):
        for j in range (0,8):
            M1,a1=load(data[:,:,j,i])
            weight[:,:,:,:,j,i],weightt[:,:,:,:,j,i]=train_cov(data[:,:,j,i])
            if i%5==1:
                print(i)
                np.save('weight8_i.npy',weight)
                np.save('weightt8_i.npy',weightt)
    np.save('weight8.npy',weight)
    np.save('weightt8.npy',weightt)
    return weight,weightt