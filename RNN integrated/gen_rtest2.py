import numpy as np
def rtest(N,ans):
    data=np.zeros((6,40,130))
    an=np.zeros((6,2))
    rtest_all=np.load('Data/'+N+'rtest.npy')
    for i in range (0,6):
        rtest=rtest_all[:,:,i].T
        data[i,:,:]=rtest[0:40,:]
        if i==ans:
            an[i,:]=[1,0]
        else:
            an[i,:]=[0,1]
    return data,an
        