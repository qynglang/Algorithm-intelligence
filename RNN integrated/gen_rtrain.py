import random
import numpy as np
def rtrain(N,a):
    data=np.zeros((128,40,130))
    ans=np.zeros((128,2))
    for i in range (0,128):
        if random.randint(0,1)==0:
            rtrain_all=np.load('Data/'+N+'rtrain.npy')
            a=random.randint(0,34)
            rtrain=rtrain_all[:,a:a+40].T
            data[i,:,:]=rtrain
            ans[i,:]=[1,0]
        else:
            #NN=['0','11','27','35','43','51','59','67','75','82','90','98']
            #n=random.randint(0,len(NN)-1)
            rtrain_all=np.load('Data/'+N+'rtrain.npy')
            #rtrain_v=np.load('Data/'+NN[n]+'rtrain.npy')
            rtrain_v=np.load('Data/'+N+'pic.npy')
            ii=random.randint(0,5)
            while ii==a:
                ii=random.randint(0,5)
            #a=random.randint(0,34)
            rtrain=rtrain_all[:,a:a+40].T
            b=random.randint(0,90)
            rtrain[:,b:b+40]=rtrain_v[5:45,10:50,ii].T
            data[i,:,:]=rtrain
            ans[i,:]=[0,1]
    return data,ans

def rref(N):
    data=np.zeros((1,40,130))
    ans=np.zeros((1,2))
    for i in range (0,1):
        rtrain_all=np.load('Data/'+N+'rtrain.npy')
        a=random.randint(0,34)
        rtrain=rtrain_all[:,a:a+40].T
        data[i,:,:]=rtrain
        ans[i,:]=[1,0]
    return data,ans
