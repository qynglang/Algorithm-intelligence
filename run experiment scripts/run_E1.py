from con1 import train_cov
#from an import ans_cut
import numpy as np
#p,pic,a=ans_cut('43')
from forw import forward,compare
from gen_mix_m import mix_m,mix_m1,mix_m2,mix_m3,mix_m4
from p_gen import p_gn
def runE(N,ans):
    pic=np.load('Data/'+N+'pic.npy')
    p=p_gn(N)
    pic1=mix_m(ans,pic)
    pic2=mix_m1(ans,pic)
    pic3=mix_m2(ans,pic)
    pic4=mix_m3(ans,pic)
    pic5=mix_m4(ans,pic)
    i=0
    W1=np.zeros((10,10,1,3,10))
    W2=np.zeros((25,25,3,6,10))
    E1=np.zeros((10,6))
    E2=np.zeros((10,6))
    E1_1=np.zeros((10,6))
    E2_1=np.zeros((10,6))
    E1_2=np.zeros((10,6))
    E2_2=np.zeros((10,6))
    E1_3=np.zeros((10,6))
    E2_3=np.zeros((10,6))
    E1_4=np.zeros((10,6))
    E2_4=np.zeros((10,6))
    while i<10:
        w1,w2,Num=train_cov(pic[:,:,:])
        if Num==1:
            W1[:,:,:,:,i]=w1
            W2[:,:,:,:,i]=w2
            I1,II2=forward(p,pic1,w1,w2)
            I_1,II2_1=forward(p,pic2,w1,w2)
            I_2,II2_2=forward(p,pic3,w1,w2)
            I_3,II2_3=forward(p,pic4,w1,w2)
            I_4,II2_4=forward(p,pic5,w1,w2)
            E1[i,:]=compare(I1,II2)
            E1_1[i,:]=compare(I_1,II2_1)
            E1_2[i,:]=compare(I_2,II2_2)
            E1_3[i,:]=compare(I_3,II2_3)
            E1_4[i,:]=compare(I_4,II2_4)
            i+=1
    E_min_n=np.zeros((50))
    for i in range (0,10):
        E_min_n[i]=np.argmin(E1[i])
    for i in range (10,20):
        E_min_n[i]=np.argmin(E1_1[i-10])
    for i in range (20,30):
        E_min_n[i]=np.argmin(E1_2[i-20])
    for i in range (30,40):
        E_min_n[i]=np.argmin(E1_1[i-30])
    for i in range (40,50):
        E_min_n[i]=np.argmin(E1_1[i-40])
    print(E_min_n[E_min_n==ans].size)
    print(np.average(E1,axis=0))
    print(np.average(E1_1,axis=0))
    print(np.average(E1_2,axis=0))
    print(np.average(E1_3,axis=0))
    print(np.average(E1_4,axis=0))
    a=np.average(E1,axis=0)[ans]/E1.mean()
    b=np.average(E1_1,axis=0)[ans]/E1_1.mean()
    c=np.average(E1_2,axis=0)[ans]/E1_2.mean()
    d=np.average(E1_3,axis=0)[ans]/E1_3.mean()
    e=np.average(E1_4,axis=0)[ans]/E1_4.mean()
    print((a+b+c+d+e)/5)
    np.save('Data/'+N+str('W1_distributed')+'.npy',W1)
    np.save('Data/'+N+str('W2_distributed')+'.npy',W2)
    np.save('Data/'+N+str('E1_distributed')+'.npy',E1)
    np.save('Data/'+N+str('E1_1_distributed')+'.npy',E1_1)
    np.save('Data/'+N+str('E1_2_distributed')+'.npy',E1_2)
    np.save('Data/'+N+str('E1_3_distributed')+'.npy',E1_3)
    np.save('Data/'+N+str('E1_4_distributed')+'.npy',E1_4)