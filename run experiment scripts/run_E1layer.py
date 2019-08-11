from con1 import train_cov
#from an import ans_cut
import numpy as np
#p,pic,a=ans_cut('43')
from forw1stlayer import forward,compare
from gen_mix_m import mix_m
def runE(N,ans):
    pic=np.load('Data/'+N+'pic.npy')
    p=np.load('Data/'+N+'p.npy')
    pic1=mix_m(ans,pic)
    i=0
    W1=np.zeros((10,10,1,3,50))
    W2=np.zeros((25,25,3,6,50))
    E1=np.zeros((50,6))
    E2=np.zeros((50,6))
    while i<50:
        w1,w2,Num=train_cov(pic[:,:,:])
        if Num==1:
            W1[:,:,:,:,i]=w1
            W2[:,:,:,:,i]=w2
            I1,II2=forward(p,pic1,w1,w2)
            E1[i,:],E2[i,:]=compare(I1,II2)
            i+=1
    E_min_n=np.zeros((50))
    E1_min_n=np.zeros((50))
    for i in range (0,50):
        E_min_n[i]=np.argmin(E1[i])
        E1_min_n[i]=np.argmin(np.abs(E2[i]))
    print(E_min_n[E_min_n==ans].size)
    print(E1_min_n[E1_min_n==ans].size)
    print(np.average(E1,axis=0))
    print(np.argsort(np.average(E1,axis=0)))
    print(np.average(E2,axis=0))
    print(np.argsort(np.average(E1,axis=0)))
    print(np.average(E1+E2,axis=0))
    print(np.argsort(np.average(E1+E2,axis=0)))
    print(E1[ans]/E1.mean())
    print(E2[ans]/np.abs(E2).mean())
    print((E1[ans]+E2[ans])/np.abs(E2+E1).mean())
    np.save('Data/'+N+str('W1_all')+'.npy',W1)
    np.save('Data/'+N+str('W2_all')+'.npy',W2)
    np.save('Data/'+N+str('E1_all')+'.npy',E1)
    np.save('Data/'+N+str('E2_all')+'.npy',E2)
    np.save('Data/'+N+str('E_average_all')+'.npy',np.average(E1,axis=0))