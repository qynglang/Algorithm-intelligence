from con1 import train_cov
import numpy as np
#import matplotlib.pyplot as plt
from fcon_1 import train_rcon
from data_gene import forward
def foward_error (n,m,ans,NN):
    data=np.load('data.npy')
    #from loadone1 import load_sample as load
    w1=np.zeros((5,5,1,3,5))
    w2=np.zeros((5,5,3,6,5))
    wf1=np.zeros((7*7*2*3, 1024,5))
    bf1=np.zeros((1024,5))
    T=0
    #m=np.array([0,3,6])
    #M12,a12=load(data[:,:,m,12])
    i=0
    while i<5:
        w,ww,Num=train_cov(data[:,:,m,n],200,3,8)
        if Num==1:
            w1[:,:,:,:,i]=w
            w2[:,:,:,:,i]=ww
            i+=1
    for i in range (0,5):
        wf1[:,:,i],bf1[:,i],Num=train_fcon(data[:,:,:,n],w1[:,:,:,:,i],w2[:,:,:,:,i],50,8)
        if Num==1:
            T=T+1
            

    EE=np.zeros((8,5))
    for j in range (0,5):
        II=forward(data[:,:,:,n],w1[:,:,:,:,j],w2[:,:,:,:,j])
        error01=II[1,:]-II[0,:]
        error12=II[2,:]-II[1,:]
        error34=II[4,:]-II[3,:]
        error54=II[5,:]-II[4,:]
        error67=II[7,:]-II[6,:]
        error78=np.zeros((8,294))
        for i in range (0,8):
            error78[i,:]=II[8+i,:]-II[7,:]
        O1=np.dot(error67.reshape(1,294),wf1[:,:,j])+bf1[:,j]
        O2=np.zeros((8,1024))
        for i in range (0,8):
            O2[i,:]=np.dot(error78[i,:].reshape(1,294),wf1[:,:,j])+bf1[:,j]
        E=np.zeros(8)
        for i in range (0,8):
            E[i]=abs(O2[i,:]-O1).sum()
        EE[:,j]=E
        N=np.zeros(5)
        for i in range (0,5):
            N[i]=np.argmin(EE[:,i])
    print(np.argmin(np.average(EE,axis=1)))
    print(np.argsort(np.average(EE,axis=1)))
    print(np.average(EE,axis=1)[ans]/np.average(EE,axis=1).mean())
    print(N)
    print(T)
    np.save('Data/'+str(NN)+'weight1_no_rotation_rerun.npy',w1)
    np.save('Data/'+str(NN)+'weight2_no_rotation_rerun.npy',w2)
    np.save('Data/'+str(NN)+'weightf_no_rotation_rerun.npy',wf1)
    np.save('Data/'+str(NN)+'biasf_no_rotation_rerun.npy',bf1)
    np.save('Data/'+str(NN)+'Error_no_rotation_rerun.npy',EE)
    np.save('Data/'+str(NN)+'Termination_no_rotation_rerun.npy',T)