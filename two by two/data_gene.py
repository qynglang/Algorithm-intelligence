import numpy as np
from convo import conv2d,maxpool2d,conv_net
import tensorflow as tf
from prepare import prep
import random
import tensorflow as tf
from convo import conv2d, maxpool2d
import numpy as np
def forward(pic,w1,w2):
    II=np.zeros((9,180))
    with tf.Session() as sess:
        for i in range (0,9):
            p1=tf.reshape(pic[:,:,i].astype('float64').reshape(1,40*50)/255 , shape=[-1, 40,50 , 1])
            I=conv2d(p1, w1)
            I=maxpool2d(I,k=3)
            I1 = conv2d(I, w2)
            I1=maxpool2d(I1,k=3)
            I1=tf.reshape(I1,[-1,180]).eval()
            II[i,:]=I1
    return II

def g_error(pic,w1,w2):
    II=forward(pic,w1,w2)
    error01=II[1,:]-II[0,:]
    error02=II[2,:]-II[0,:]
    data=np.zeros((8,180))
    answer=np.zeros((8,2))
    for i in range (0,8):
        r=random.randint(0,1)
        if r==1:
            data[i,:]=error01.reshape(1,180)
            answer[i,:]=[1,0]
        elif r==0:
            data[i,:]=error02.reshape(1,180)
            answer[i,:]=[0,1]
    return data, answer

