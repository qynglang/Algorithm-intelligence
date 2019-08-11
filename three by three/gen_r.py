import numpy as np
import random
def gen_r(i):
    t=np.zeros((128,13))
    a=np.zeros((128,7))
    train=np.zeros((7,13))
    ans=np.zeros((7,7))
    data=np.load('data.npy')
    q=data[data[:,0]==i]
    q=q[:,2:]
    train[0,:]=q[1,:]-q[0,:]
    train[1,:]=q[2,:]-q[1,:]
    train[2,:]=q[2,:]-q[0,:]
    train[3,:]=q[4,:]-q[3,:]
    train[4,:]=q[5,:]-q[4,:]
    train[5,:]=q[5,:]-q[3,:]
    train[6,:]=q[7,:]-q[6,:]
    ans[0,:]=[1,0,0,0,0,0,0]
    ans[1,:]=[0,1,0,0,0,0,0]
    ans[2,:]=[0,0,1,0,0,0,0]
    ans[3,:]=[0,0,0,1,0,0,0]
    ans[4,:]=[0,0,0,0,1,0,0]
    ans[5,:]=[0,0,0,0,0,1,0]
    ans[6,:]=[0,0,0,0,0,0,1]
    
    for i in range (0,128):
        k=random.randint(0,7)
        if k==7:
            t[i,:]=np.random.randint(5,size=13)
            a[i,:]=[0,0,0,0,0,0,0]
        else:
            t[i,:]=train[k,:]
            a[i,:]=ans[k,:]
    return t,a
def gen_r_test(i,ans):
    test=np.zeros((16,13))
    a=np.zeros((16,7))
    data=np.load('data.npy')
    q=data[data[:,0]==i]
    q=q[:,2:]
    test[0,:]=q[8,:]-q[7,:]
    test[1,:]=q[8,:]-q[6,:]
    test[2,:]=q[9,:]-q[7,:]
    test[3,:]=q[9,:]-q[6,:]
    test[4,:]=q[10,:]-q[7,:]
    test[5,:]=q[10,:]-q[6,:]
    test[6,:]=q[11,:]-q[7,:]
    test[7,:]=q[11,:]-q[6,:]
    test[8,:]=q[12,:]-q[7,:]
    test[9,:]=q[12,:]-q[6,:]
    test[10,:]=q[13,:]-q[7,:]
    test[11,:]=q[13,:]-q[6,:]
    test[12,:]=q[14,:]-q[7,:]
    test[13,:]=q[14,:]-q[6,:]
    test[14,:]=q[15,:]-q[7,:]
    test[15,:]=q[15,:]-q[6,:]
    for i in range (0,16):
        a[i,:]=[0,0,0,0,0,0,0]

    return test,a

def gen_r_ref(i):

    train=np.zeros((7,13))
    ans=np.zeros((7,7))
    data=np.load('data.npy')
    q=data[data[:,0]==i]
    q=q[:,2:]
    train[0,:]=q[1,:]-q[0,:]
    train[1,:]=q[2,:]-q[1,:]
    train[2,:]=q[2,:]-q[0,:]
    train[3,:]=q[4,:]-q[3,:]
    train[4,:]=q[5,:]-q[4,:]
    train[5,:]=q[5,:]-q[3,:]
    train[6,:]=q[7,:]-q[6,:]
    ans[0,:]=[1,0,0,0,0,0,0]
    ans[1,:]=[0,1,0,0,0,0,0]
    ans[2,:]=[0,0,1,0,0,0,0]
    ans[3,:]=[0,0,0,1,0,0,0]
    ans[4,:]=[0,0,0,0,1,0,0]
    ans[5,:]=[0,0,0,0,0,1,0]
    ans[6,:]=[0,0,0,0,0,0,1]
    return train,ans
        

    
    
            
    

    