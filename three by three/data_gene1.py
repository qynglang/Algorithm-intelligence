import numpy as np
from convo import conv2d,maxpool2d,conv_net
import tensorflow as tf
from prepare import prep
import random
import tensorflow as tf
from convo import conv2d, maxpool2d
import numpy as np
def forward(pic,w1,w2):
    II=np.zeros((16,6*7*7))
    with tf.Session() as sess:
        for i in range (0,16):
            p1=tf.reshape(pic[:,:,i].astype('float64').reshape(1,28*28)/255 , shape=[-1, 28,28 , 1])
            I=conv2d(p1, w1)
            I=maxpool2d(I,k=2)
            I1 = conv2d(I, w2)
            I1=maxpool2d(I1,k=2)
            I1=tf.reshape(I1,[-1,6*7*7]).eval()
            II[i,:]=I1
    return II

def g_error(pic,w1,w2):
    II=forward(pic,w1,w2)
    error03=II[3,:]-II[0,:]
    error36=II[6,:]-II[3,:]
    error41=II[4,:]-II[1,:]
    error52=II[5,:]-II[2,:]
    data=np.zeros((8,49*2*3))
    answer=np.zeros((8,4))
    for i in range (0,8):
        r=random.randint(0,4)
        if r==1:
            data[i,:]=error03.reshape(1,49*2*3)
            answer[i,:]=[1,0,0,0]
        elif r==0:
            data[i,:]=error36.reshape(1,49*2*3)
            answer[i,:]=[0,1,0,0]
        elif r==0:
            data[i,:]=error41.reshape(1,49*2*3)
            answer[i,:]=[0,0,1,0]
        elif r==0:
            data[i,:]=error52.reshape(1,49*2*3)
            answer[i,:]=[0,0,0,1]
    return data, answer

