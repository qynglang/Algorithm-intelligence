import numpy as np
#import matplotlib.pyplot as plt
from convo import conv2d,maxpool2d,conv_net
import tensorflow as tf
from prepare import prep
from pic_n import pic3,pic3_2
from t import gentest
from fcon import train_fcon
from weight import wei
#from con import train_cov
data=np.load('data.npy')
weight=np.load('weight8.npy')
weightt=np.load('weightt8.npy')
#weight,weightt=wei()
answer=np.load('ans.npy')
#M,A=pic3(data[:,:,:,0].reshape(28,28,16),weight,weightt)
#T,An=gentest(data[:,:,:,0].reshape(28,28,16),weight,weightt,answer[0])
ll=np.zeros((8,2,36))
pp=np.zeros((8,2,36))
for i in range (0,36):
    ll[:,:,i],pp[:,:,i]=train_fcon(data[:,:,:,i].reshape(28,28,16),weight[:,:,:,:,:,i],weightt[:,:,:,:,:,i],answer[i])
    if i%5==0:
        np.save('ll8.npy',ll)
        np.save('pp8.npy',pp)
    np.save('ll8.npy',ll)
    np.save('pp8.npy',pp)
    