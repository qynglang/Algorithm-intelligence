from con1 import train_cov
#from an import ans_cut
import numpy as np
#p,pic,a=ans_cut('43')
from forw import forward,compare
from gen_mix_m import mix_m
N='27'
ans=0
pic=np.load('Data/'+N+'pic.npy')
p=np.load('Data/'+N+'p.npy')
pic1=mix_m(1,pic)
i=0
W1=np.zeros((10,10,1,3,50))
W2=np.zeros((25,25,3,6,50))
E1=np.zeros((50,6))
E2=np.zeros((50,6))
m=[0,1,2]
while i<50:
    w1,w2,Num=train_cov(pic[:,:,m])
    if Num==1:
        W1[:,:,:,:,i]=w1
        W2[:,:,:,:,i]=w2
        I1,II2=forward(p,pic1,w1,w2)
        E1[i,:]=compare(I1,II2)
        i+=1
E_min_n=np.zeros((50))
#E1_min_n=np.zeros((50))
for i in range (0,50):
    E_min_n[i]=np.argmin(E1[i])
    #E1_min_n[i]=np.argmin(np.abs(E2[i]))
print(E_min_n[E_min_n==ans].size)
#print(E1_min_n[E1_min_n==ans].size)
print(np.average(E1,axis=0))
#print(np.average(E2,axis=0))
print(np.average(E1,axis=0)[ans]/E1.mean())
#print(np.abs(E2).mean())
np.save('Data/'+N+str('W1_c')+'.npy',W1)
np.save('Data/'+N+str('W2_c')+'.npy',W2)
np.save('Data/'+N+str('E1_c')+'.npy',E1)        