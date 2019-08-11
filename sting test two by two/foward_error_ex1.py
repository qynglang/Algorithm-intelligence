from con1 import train_cov
import numpy as np
#import matplotlib.pyplot as plt
from fcon_excape_c import train_fcon
from fcon_excape_c2 import train_fcon1
from fcon_excape_c3 import train_fcon2
from data_gene import forward
def foward_error (n,m,ans,N_str):
    tr=5
    data=np.load('data_4.npy')
    #from loadone1 import load_sample as load
    w1=np.zeros((5,5,1,3,tr))
    w2=np.zeros((5,5,3,6,tr))
    wf1=np.zeros((180, 1024,tr))
    bf1=np.zeros((1024,tr))
    wf2=np.zeros((1204, 1024,tr))
    bf2=np.zeros((1024,tr))
    wf3=np.zeros((1204, 1024,tr))
    bf3=np.zeros((1024,tr))
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
        wf1[:,:,i],bf1[:,i],l,Num=train_fcon(data[:,:,:,n],w1[:,:,:,:,i],w2[:,:,:,:,i],50,8)
        if Num==1:
            T=T+1
    for i in range (0,tr):
        wf2[:,:,i],bf2[:,i],l1,Num=train_fcon1(data[:,:,:,n],w1[:,:,:,:,i],w2[:,:,:,:,i],50,8,l)
        if Num==1:
            T=T+1
    for i in range (0,tr):
        wf3[:,:,i],bf3[:,i],l2,Num=train_fcon2(data[:,:,:,n],w1[:,:,:,:,i],w2[:,:,:,:,i],50,8,l1)
        if Num==1:
            T=T+1
    print(l1.shape)
#     EE=np.zeros((6,5))
#     EE2=np.zeros((6,5))
#     for j in range (0,5):
#         II=forward(data[:,:,:,n],w1[:,:,:,:,j],w2[:,:,:,:,j])
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
#         N=np.zeros(5)
#         for i in range (0,5):
#             N[i]=np.argmin(EE[:,i])
#         NN=np.zeros(5)
#         for i in range (0,5):
#             NN[i]=np.argmin(EE2[:,i])
#         NNN=np.zeros(5)
#         for i in range (0,5):
#             NNN[i]=np.argmin(EE2[:,i]+EE[:,i])
#     print('x direction error:',np.argsort(np.average(EE,axis=1)))
#     print('y direction error:',np.argsort(np.average(EE2,axis=1)))
#     print('normalized correct x direction error:',np.average(EE,axis=1)[ans]/np.average(EE,axis=1).mean())
#     print('normalized correct y direction error:',np.average(EE2,axis=1)[ans]/np.average(EE2,axis=1).mean())
#     print('sum error:',np.argsort(np.average(EE2+EE,axis=1)))
#     print('normalized correct sum error:',np.average(EE2+EE,axis=1)[ans]/np.average(EE+EE2,axis=1).mean())
#     print('number of correct trials, x:',N)
#     print('number of correct trials, y:',NN)
#     print(T)
#     np.save('Data/'+str(N_str)+'weight1.npy',w1)
#     np.save('Data/'+str(N_str)+'weight2.npy',w2)
#     np.save('Data/'+str(N_str)+'weightf.npy',wf1)
#     np.save('Data/'+str(N_str)+'biasf.npy',bf1)
#     np.save('Data/'+str(N_str)+'Error.npy',EE)
#     np.save('Data/'+str(N_str)+'Error.npy',EE2)
#     np.save('Data/'+str(N_str)+'Termination.npy',T)
    np.save('Data/'+str(N_str)+'weight1c_ex.npy',w1)
    np.save('Data/'+str(N_str)+'weight2c_ex.npy',w2)
    np.save('Data/'+str(N_str)+'weightf1_ex.npy',wf1)
    np.save('Data/'+str(N_str)+'biasf1_ex.npy',bf1)
    np.save('Data/'+str(N_str)+'weightf2_ex.npy',wf2)
    np.save('Data/'+str(N_str)+'biasf2_ex.npy',bf2)
    np.save('Data/'+str(N_str)+'weightf3_ex.npy',wf3)
    np.save('Data/'+str(N_str)+'biasf3_ex.npy',bf3)