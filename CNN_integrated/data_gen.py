import random
import tensorflow as tf
from convo import conv2d, maxpool2d
import numpy as np
from p_gen import p_gn
def g_error(pic,w1,w2,ans,N):
    data=np.zeros((6,36))
    a=np.zeros((6,2))
    for i in range (0,6):
        if random.randint(0,1)==1:
            p=p_gn(N)
            p1=tf.reshape(p.astype('float64').reshape(1,3000)/255 , shape=[-1, 50,60 , 1])
            II=conv2d(p1, w1)
            II=maxpool2d(II,k=5)
            II1 = conv2d(II, w2)
            II1=maxpool2d(II1,k=5)
            II1=tf.reshape(II1,[-1,36]).eval()
            data[i,:]=II1
            a[i,:]=[1,0]
        if random.randint(0,1)==0:
            k=random.randint(0,5)
            while k==ans:
                k=random.randint(0,5)
            p=pic[:,:,k]
            p1=tf.reshape(p.astype('float64').reshape(1,3000)/255 , shape=[-1, 50,60 , 1])
            II=conv2d(p1, w1)
            II=maxpool2d(II,k=5)
            II1 = conv2d(II, w2)
            II1=maxpool2d(II1,k=5)
            II1=tf.reshape(II1,[-1,36]).eval()
            data[i,:]=II1
            a[i,:]=[0,1]
            
    return data,a