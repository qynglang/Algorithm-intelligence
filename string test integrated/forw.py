import tensorflow as tf
from convo import conv2d, maxpool2d
import numpy as np
def forward(p,pic,w1,w2):
    with tf.Session() as sess:
        p1=tf.reshape(p.astype('float64').reshape(1,3000)/255 , shape=[-1, 50,60 , 1])
        I=conv2d(p1, w1)
        I=maxpool2d(I,k=5)
        I1 = conv2d(I, w2)
        I1=maxpool2d(I1,k=5)
        I1=tf.reshape(I1,[-1,36]).eval()
    II2=np.zeros((6,36))
    with tf.Session() as sess:
        for i in range (0,6):
            p1=tf.reshape(pic[:,:,i].astype('float64').reshape(1,3000)/255 , shape=[-1, 50,60 , 1])
            II=conv2d(p1, w1)
            II=maxpool2d(II,k=5)
            II1 = conv2d(II, w2)
            II1=maxpool2d(II1,k=5)
            II1=tf.reshape(II1,[-1,36]).eval()
            II2[i,:]=II1
    return I1,II2

def compare(I1,II2):        
    E=np.zeros(6)
    for i in range (0,6):
        E[i]=np.abs(II2[i,:]-I1).sum()/np.abs(I1).sum()
    return E


    