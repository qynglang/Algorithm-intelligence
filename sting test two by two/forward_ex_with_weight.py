import numpy as np

from data_gene import forward
def foward_error (n,m,ans,N_str):
    data=np.load('data_4.npy')
    w1=np.load('Data/'+str(N_str)+'weight1c_ex.npy')
    w2=np.load('Data/'+str(N_str)+'weight2c_ex.npy')
    wf1=np.load('Data/'+str(N_str)+'weightf1_ex.npy')
    bf1=np.load('Data/'+str(N_str)+'biasf1_ex.npy')
    wf2=np.load('Data/'+str(N_str)+'weightf2_ex.npy')
    bf2=np.load('Data/'+str(N_str)+'biasf2_ex.npy')
    wf3=np.load('Data/'+str(N_str)+'weightf3_ex.npy')
    bf3=np.load('Data/'+str(N_str)+'biasf3_ex.npy')
    O2=np.zeros((6,1024))
    O3=np.zeros((6,1024))
    O4=np.zeros((6,1024))
    E=np.zeros(6)
    EE=np.zeros((6,w1.shape[4]))
    E_1=np.zeros(6)
    EE_1=np.zeros((6,w1.shape[4]))
    E_2=np.zeros(6)
    EE_2=np.zeros((6,w1.shape[4]))
    for j in range (0,w1.shape[4]):
        II=forward(data[:,:,:,n],w1[:,:,:,:,j],w2[:,:,:,:,j])
        error01=II[1,:]-II[0,:]
        error02=II[2,:]-II[0,:]
        error23=np.zeros((6,180))
        error13=np.zeros((6,180))
        for i in range (1,7):
            error23[i-1,:]=II[2+i,:]-II[2,:]
        for i in range (1,7):
            error13[i-1,:]=II[2+i,:]-II[1,:]
            
        O1=np.dot(error01.reshape(1,180),wf1[:,:,j])+bf1[:,j]
        for i in range (0,6):
            O2[i,:]=np.dot(error23[i,:].reshape(1,180),wf1[:,:,j])+bf1[:,j]
        O1_2=np.dot(np.hstack((error01,O1.reshape(1024))).reshape(1,1204),wf2[:,:,j])+bf2[:,j]

        
        for i in range (0,6):
            #print((np.dot(np.hstack((error23[i,:],O2[i,:].reshape(1024))).reshape(1,1204),wf2[:,:,j])+bf2[:,j]).shape)
            O3[i,:]=np.dot(np.hstack((error23[i,:],O2[i,:].reshape(1024))).reshape(1,1204),wf2[:,:,j])+bf2[:,j]
        
        O1_3=np.dot(np.hstack((error01,O1_2.reshape(1024))).reshape(1,1204),wf3[:,:,j])+bf3[:,j]
        for i in range (0,6):
            O4[i,:]=np.dot(np.hstack((error23[i,:],O3[i,:].reshape(1024))).reshape(1,1204),wf3[:,:,j])+bf3[:,j]
        for i in range (0,6):
            E[i]=abs(O2[i,:]-O1).sum()
        EE[:,j]=E
        for i in range (0,6):
            E_1[i]=abs(O3[i,:]-O1_2).sum()
        EE_1[:,j]=E_1
        for i in range (0,6):
            E_2[i]=abs(O4[i,:]-O1_3).sum()
        EE_2[:,j]=E_2
        
    E_min_n=np.zeros((5))
    for i in range (0,w1.shape[4]):
        E_min_n[i]=np.argmin(EE[:,i])
    #print(EE)
    print(E_min_n[E_min_n==ans].size)
    #print(E1_min_n[E1_min_n==ans].size)
    print(np.average(EE,axis=1))
    print(np.argsort(np.average(EE,axis=1)))
    #print(np.average(E2,axis=0))
    print(np.average(EE,axis=1)[ans]/EE.mean())
    
    E_min_n1=np.zeros((5))
    for i in range (0,w1.shape[4]):
        E_min_n1[i]=np.argmin(EE_1[:,i])
    print(E_min_n1[E_min_n1==ans].size)
    #print(E1_min_n[E1_min_n==ans].size)
    print(np.average(EE_1,axis=1))
    print(np.argsort(np.average(EE_1,axis=1)))
    #print(np.average(E2,axis=0))
    print(np.average(EE_1,axis=1)[ans]/EE_1.mean())
    
    E_min_n2=np.zeros((5))
    for i in range (0,w1.shape[4]):
        E_min_n2[i]=np.argmin(EE_2[:,i])
    print(E_min_n2[E_min_n2==ans].size)
    #print(E1_min_n[E1_min_n==ans].size)
    print(np.average(EE_2,axis=1))
    print(np.argsort(np.average(EE_2,axis=1)))
    #print(np.average(E2,axis=0))
    print(np.average(EE_2,axis=1)[ans]/EE_2.mean())