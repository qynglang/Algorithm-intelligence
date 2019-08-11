import numpy as np
def rtest(N,ans):
    data=np.zeros((12,20,130))
    an=np.zeros((12,2))
    rtest_all=np.load('Data/'+N+'rtest.npy')
    for i in range (0,6):
        rtest=rtest_all[:,:,i].T
        data[i*2,:,:]=rtest[0:20,:]
        data[i*2+1,:,:]=rtest[20:40,:]
        if i==ans:
            an[i*2,:]=[1,0]
            an[i*2+1,:]=[1,0]
        else:
            an[i*2,:]=[0,1]
            an[i*2+1,:]=[0,1] 
    return data,an
        
    
    