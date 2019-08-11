
from con1 import train_cov
import numpy as np
#import matplotlib.pyplot as plt
from rcon import train_rcon
from data_gene import forward

def foward_error (n,m,ans,N_str):
    tr=50
    EEE=np.zeros((6,tr))
    NN=np.zeros(tr)
    data=np.load('data_4.npy')
    #from loadone1 import load_sample as load
    w1=np.zeros((5,5,1,3,tr))
    w2=np.zeros((5,5,3,6,tr))
    wf1=np.zeros((128, 2,tr))
    bf1=np.zeros((2,tr))
    T=0
    #m=np.array([0,3,6])
    #M12,a12=load(data[:,:,m,12])
    i=0
    while i<tr:
        w,ww,Num=train_cov(data[:,:,m,n],200,3,8)
        if Num==1:
            w1[:,:,:,:,i]=w
            w2[:,:,:,:,i]=ww
            i+=1

    for i in range (0,tr):
        EE,N,Num=train_rcon(data[:,:,:,n],w1[:,:,:,:,i],w2[:,:,:,:,i],100,8,n)
        if Num==1:
            T=T+1
        EEE[:,i]=EE.reshape(6)
        NN[i]=N

#     #EE2=np.zeros((6,tr))
#     for j in range (0,tr):
#         II=forward(data[:,:,:,n],w1[:,:,:,:,j],w2[:,:,:,:,j])
#         data=np.zeros((1,8,180))
#         data[:,0,:]=II[0,:].reshape(1,180)
#         data[:,1,:]=II[1,:].reshape(1,180)
#         data[:,2,:]=II[2,:].reshape(1,180)
#         data[:,4,:]=II[0,:].reshape(1,180)
#         data[:,5,:]=II[2,:].reshape(1,180)
#         data[:,6,:]=II[1,:].reshape(1,180)
#         for i in range (1,7):
#             III=II[2+i,:].reshape(1,180)
#             data[:,3,:]=III.reshape(1,180)
#             data[:,7,:]=III.reshape(1,180)
#             O1=RNN(data[:,0:2,:],wf1[:,:,j],bf1[:,j])
#             O2=RNN(data[:,2:4,:],wf1[:,:,j],bf1[:,j])
#             O3=RNN(data[:,4:6,:],wf1[:,:,j],bf1[:,j])
#             O4=RNN(data[:,6:8,:],wf1[:,:,j],bf1[:,j])
#             EE[i,j]=np.abs(O2-O1).sum()+np.abs(O4-O3).sum()
#         N=np.zeros(tr)
#         for i in range (0,tr):
#             N[i]=np.argmin(EE[:,i])        
#         error01=II[1,:]-II[0,:]
#         error02=II[2,:]-II[0,:]
#         error23=np.zeros((6,180))
#         error13=np.zeros((6,180))
#         for i in range (1,7):
#             error23[i-1,:]=II[2+i,:]-II[2,:]
#         for i in range (1,7):
#             error13[i-1,:]=II[2+i,:]-II[1,:]
#         O1=np.dot(error01.reshape(1,180),wf1[:,:,j])+bf1[:,j]
#         O2=np.zeros((6,1024))
#         for i in range (0,6):
#             O2[i,:]=np.dot(error23[i,:].reshape(1,180),wf1[:,:,j])+bf1[:,j]
#         E=np.zeros(6)
#         for i in range (0,6):
#             E[i]=abs(O2[i,:]-O1).sum()
#         EE[:,j]=E
#         O3=np.dot(error02.reshape(1,180),wf1[:,:,j])+bf1[:,j]
#         O4=np.zeros((6,1024))
#         for i in range (0,6):
#             O4[i,:]=np.dot(error13[i,:].reshape(1,180),wf1[:,:,j])+bf1[:,j]
#         EEE=np.zeros(6)
#         for i in range (0,6):
#             EEE[i]=abs(O4[i,:]-O3).sum()
#         EE2[:,j]=EEE
#         N=np.zeros(tr)
#         for i in range (0,tr):
#             N[i]=np.argmin(EE[:,i])
#         NN=np.zeros(tr)
#         for i in range (0,tr):
#             NN[i]=np.argmin(EE2[:,i])
#         NNN=np.zeros(tr)
#         for i in range (0,tr):
#             NNN[i]=np.argmin(EE2[:,i]+EE[:,i])
    print(np.argsort(np.average(EEE,axis=1)))
#     print('y direction error:',np.argsort(np.average(EE2,axis=1)))
    print(np.average(EE,axis=1)[ans]/np.average(EEE,axis=1).mean())
#     print('normalized correct y direction error:',np.average(EE2,axis=1)[ans]/np.average(EE2,axis=1).mean())
#     print('sum error:',np.argsort(np.average(EE2+EE,axis=1)))
#     print('normalized correct sum error:',np.average(EE2+EE,axis=1)[ans]/np.average(EE+EE2,axis=1).mean())
    print(np.where(np.array(NN)==ans)[0].size)
#     print('number of correct trials, y:',NN)
#     print(T)
    np.save('Data/'+str(N_str)+'weight1_r.npy',w1)
    np.save('Data/'+str(N_str)+'weight2_r.npy',w2)
#     np.save('Data/'+str(N_str)+'weightf_r.npy',wf1)
#     np.save('Data/'+str(N_str)+'biasf_r.npy',bf1)
    np.save('Data/'+str(N_str)+'Error.npy',EEE)
#     np.save('Data/'+str(N_str)+'Error.npy',EE2)
#     np.save('Data/'+str(N_str)+'Termination.npy',T)