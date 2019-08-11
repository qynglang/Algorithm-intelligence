from con1 import train_cov
#from an import ans_cut
import numpy as np
#p,pic,a=ans_cut('43')
from forw_f import forward,compare
from gen_mix_m import mix_m
from fcon3 import train_fcon
from p_gen_2 import p_gn
def runE(N,ans):
    tr=50
    pic=np.load('Data/'+N+'pic.npy')
    p=np.load('Data/'+N+'p.npy')
    pic1=mix_m(ans,pic)
    i=0
    W1=np.zeros((10,10,1,3,tr))
    W2=np.zeros((25,25,3,6,tr))
    W=np.zeros((36,1024,tr))
    B=np.zeros((1024,tr))
    E1=np.zeros((tr,6))
    #E2=np.zeros((tr,6))
    while i<tr:
        w1,w2,Num=train_cov(pic[:,:,:])
        if Num==1:
            W1[:,:,:,:,i]=w1
            W2[:,:,:,:,i]=w2
            i+=1
    k=0
    T=0
    while k<tr:
        print(k)
        w,b,Num=train_fcon(pic,W1[:,:,:,:,k],W2[:,:,:,:,k],50,8,ans,N)
        W[:,:,k]=w
        B[:,k]=b
        if Num==1:
            T+=1
        k+=1
    #return W,B,T
    for i in range (0,tr):
        II1,II3=forward(p,pic1,W1[:,:,:,:,i],W2[:,:,:,:,i],W[:,:,i],B[:,i])
        E1[i,:]=compare(II1,II3)
    E_min_n=np.zeros((tr))
    for i in range (0,tr):
        E_min_n[i]=np.argmin(E1[i])
    print(E_min_n[E_min_n==ans].size)
    #print(E1_min_n[E1_min_n==ans].size)
    print(np.average(E1,axis=0))
    #print(np.average(E2,axis=0))
    print(np.argsort(np.average(E1,axis=0)))
    print(np.average(E1,axis=0)[ans]/E1.mean())
    
    #print(np.abs(E2).mean())
    np.save('Data/'+N+str('W1_f')+'.npy',W1)
    np.save('Data/'+N+str('W2_f')+'.npy',W2)
    np.save('Data/'+N+str('W1_f')+'.npy',W)
    np.save('Data/'+N+str('W2_f')+'.npy',B)
    np.save('Data/'+N+str('E1_f')+'.npy',E1)
    #np.save('Data/'+N+str('E2_f')+'.npy',E2)
    #np.save('Data/'+N+str('E_average_f')+'.npy',np.average(E1,axis=0))