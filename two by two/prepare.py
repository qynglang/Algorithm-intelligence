import numpy as np
from convo import conv2d,maxpool2d,conv_net
import tensorflow as tf# from scipy.signal import convolve2d
def prep(pic,weight1,weight2):
    n=weight1.shape[4]
    M=np.zeros((n,pic.shape[2],7020))
    with tf.Session() as sess:
        p=np.zeros((pic.shape[2],3000))
        for i in range(0,pic.shape[2]):
            p[i,:]=pic[:,:,i].reshape(3000,)
        p=tf.reshape(p, shape=[-1, 50,60 , 1])
        for k in range (0,n):
            I=conv2d(p, weight1[:,:,:,:,k].reshape(25,25,1,3))
            I=maxpool2d(I,k=2)
            I1 = conv2d(I, weight2[:,:,:,:,k].reshape(50,50,3,6))
        # Max Pooling (down-sampling)
            I1 = maxpool2d(I1, k=2)
            I1=tf.reshape(I1,[-1,7020]).eval()
            M[k,:,:]=I1
    return M