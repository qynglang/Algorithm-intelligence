import numpy as np
from convo import conv2d,maxpool2d,conv_net
import tensorflow as tf# from scipy.signal import convolve2d
def prep(pic,weight1,weight2,a):
    n=weight1.shape[4]
    M=np.zeros((n,pic.shape[2],49*2*a))
    with tf.Session() as sess:
        p=np.zeros((pic.shape[2],784))
        for i in range(0,pic.shape[2]):
            p[i,:]=pic[:,:,i].reshape(784,)
        p=tf.reshape(p, shape=[-1, 28, 28, 1])
        for k in range (0,n):
            I=conv2d(p, weight1[:,:,:,:,k].reshape(5,5,1,a))
            I=maxpool2d(I,2)
            I1 = conv2d(I, weight2[:,:,:,:,k].reshape(5,5,a,2*a))
        # Max Pooling (down-sampling)
            I1 = maxpool2d(I1,2)
            I1=tf.reshape(I1,[-1,49*2*a]).eval()
            M[k,:,:]=I1
    return M