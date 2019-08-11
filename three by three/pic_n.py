import numpy as np
from convo import conv2d,maxpool2d,conv_net
import tensorflow as tf
from prepare import prep
import random
def pic3(pic,weight1,weight2,a):
    data=np.zeros((8,49*2*a*3*weight1.shape[4]))
    answer=np.zeros((8,2))
    for i in range (0,8):
        r=random.randint(0,4)
        if r==1:
            num=[0,1,2]
        elif r==0:
            num=[3,4,5]
        else:
            num=np.random.choice(8, 3)
        data[i,:]=prep(pic[:,:,num],weight1,weight2,a).reshape(1,49*2*a*3*weight1.shape[4])
        if np.array_equal(num,[0,1,2]) | np.array_equal(num,[3,4,5]):
            answer[i,:]=[1,0]
        else:
            answer[i,:]=[0,1]
    return data, answer
def pic3_2(pic,weight1,weight2,a):
    data=np.zeros((8,49*2*a*3*weight1.shape[4]))
    answer=np.zeros((8,2))
    for i in range (0,8):
        r=random.randint(0,3)
        if r==1:
            num=[0,3,6]
        elif r==0:
            num=[1,4,7]
        else:
            num=np.random.choice(8, 3)
        data[i,:]=prep(pic[:,:,num],weight1,weight2,a).reshape(1,49*2*a*3*weight1.shape[4])
        if np.array_equal(num,[0,3,6]) | np.array_equal(num,[1,4,7]):
            answer[i,:]=[1,0]
        else:
            answer[i,:]=[0,1]
    return data, answer