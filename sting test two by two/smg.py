#sequence matrices generator
import numpy as np
import random
def load(shape,bs,m=5):
    D=len(shape)
    if D==1:
        data=np.zeros((bs,shape[0]))
        answer=np.zeros((bs,10))
        for i in range (0,bs):
            M=random.randint(0,20)
            k=random.randint(1,m)
            data[i,:],answer[i,:]=globals()['generator'+str(k)](shape)
            data[i,:]+=M
    if D==2:
        data=np.zeros((bs,shape[0],shape[1]))
        answer=np.zeros((bs,10))
        for i in range (0,bs):
            M=random.randint(0,20)
            k=random.randint(1,m)
            data[i,:,:],answer[i,:]=globals()['generator'+str(k)](shape)
            data[i,:,:]+=M
    if D==3:
        data=np.zeros((bs,shape[0],shape[1],shape[2]))
        answer=np.zeros((bs,10))
        for i in range (0,bs):
            M=random.randint(0,20)
            k=random.randint(1,m)
            data[i,:,:,:],answer[i,:]=globals()['generator'+str(k)](shape)
            data[i,:,:,:]+=M
    return data,answer
def load_test(shape,bs=100,m=5):
    D=len(shape)
    if D==1:
        data=np.zeros((bs,shape[0]))
        answer=np.zeros((bs,10))
        for i in range (0,10):
            for k in range (1,11):
                    M=random.randint(20,30)
                    data[i*10+k-1,:],answer[i*10+k-1,:]=globals()['generator'+str(k)](shape)
                    data[i*10+k-1,:]+=M
    if D==2:
        data=np.zeros((bs,shape[0],shape[1]))
        answer=np.zeros((bs,10))
        for i in range (0,10):
            for k in range (1,11):
                    M=random.randint(20,30)
                    data[i*10+k-1,:,:],answer[i*10+k-1,:]=globals()['generator'+str(k)](shape)
                    data[i*10+k-1,:,:]+=M
    if D==3:
        data=np.zeros((bs,shape[0],shape[1],shape[2]))
        answer=np.zeros((bs,10))
        for i in range (0,10):
            M=random.randint(20,30)
            for k in range (1,11):
                for w in range (0,10):
                    data[i*10+w,:,:,:],answer[i*10+w,:]=globals()['generator'+str(k)](shape)
                    data[i*10+w,:,:,:]+=M
    return data,answer

#conherent
def generator1(shape):
    D=len(shape)
    if D==1:
        data=np.ones((shape[0]))
        answer=np.zeros((1,10))
        answer[:,0]=1
    if D==2:
        data=np.ones((shape[0],shape[1]))
        answer=np.zeros((1,10))
        answer[:,0]=1
    if D==3:
        data=np.ones((shape[0],shape[1],shape[2]))
        answer=np.zeros((1,10))
        answer[:,0]=1
    return data, answer

#same relationship
def generator2(shape):
    D=len(shape)
    if D==1:
        data=np.ones((shape[0]))
        d=np.full((int(shape[0]/2)+1,2), [1,2]).reshape(int(2*(int(shape[0]/2)+1)))
        data[0:shape[0]]=d[0:shape[0]]
        answer=np.zeros((1,10))
        answer[:,1]=1
    if D==2:
        data=np.ones((shape[0],shape[1]))
        d0=np.full((int(shape[0]/2)+1,2), [1,2]).reshape(int(2*(int(shape[0]/2)+1)),1)
        d1=np.full((int(shape[0]/2)+1,2), [2,1]).reshape(int(2*(int(shape[0]/2)+1)),1)

        data[0:shape[0],::2]=d0[0:shape[0]]
        data[0:shape[0],1::2]=d1[0:shape[0]]
        answer=np.zeros((1,10))
        answer[:,1]=1
    if D==3:
        data=np.ones((shape[0],shape[1],shape[2]))
        d0=np.full((int(shape[0]/2)+1,2), [1,2]).reshape(int(2*(int(shape[0]/2)+1)),1)
        d1=np.full((int(shape[0]/2)+1,2), [2,1]).reshape(int(2*(int(shape[0]/2)+1)),1)
        for i in range (0,shape[2]):
            if i%2==1:
                data[0:shape[0],::2,i]=d0[0:shape[0]]
                data[0:shape[0],1::2,i]=d1[0:shape[0]]
            else:
                data[0:shape[0],::2,i]=d1[0:shape[0]]
                data[0:shape[0],1::2,i]=d0[0:shape[0]]
        answer=np.zeros((1,10))
        answer[:,1]=1
    return data, answer

#progression
def generator3(shape):
    D=len(shape)
    if D==1:
        data=np.arange(shape[0])
        #data[:,0:shape[0]]=d[:,0:shape[0]]
        answer=np.zeros((1,10))
        answer[:,2]=1
    if D==2:
        data=np.zeros((shape[0],shape[1]))
        for i in range (0,shape[1]):
            data[:,i]=np.arange(shape[0])+i
        answer=np.zeros((1,10))
        answer[:,2]=1
    if D==3:
        data=np.zeros((shape[0],shape[1],shape[2]))
        for i in range (0,shape[1]):
            for j in range (0,shape[2]):
                data[:,i,j]=np.arange(shape[0])+i+j
        answer=np.zeros((1,10))
        answer[:,2]=1
    return data, answer

#minus
def generator4(shape):
    D=len(shape)
    if D==1:
        data=(np.rot90(np.rot90(np.arange(shape[0]).reshape(shape[0],1)))).reshape(shape[0])
        #data[:,0:shape[0]]=d[:,0:shape[0]]
        answer=np.zeros((1,10))
        answer[:,3]=1
    if D==2:
        data=np.zeros((shape[0],shape[1]))
        for i in range (0,shape[1]):
            data[:,i]=(np.rot90(np.rot90(np.arange(shape[0]).reshape(shape[0],1)))+shape[1]-i).reshape(shape[0])
        answer=np.zeros((1,10))
        answer[:,3]=1
    if D==3:
        data=np.zeros((shape[0],shape[1],shape[2]))
        for i in range (0,shape[1]):
            for j in range (0,shape[2]):
                data[:,i,j]=(np.rot90(np.rot90(np.arange(shape[0]).reshape(shape[0],1)))+shape[1]-i+shape[2]-j).reshape(shape[0])
        answer=np.zeros((1,10))
        answer[:,3]=1
    data=data-np.min(data)
    return data, answer

#permutation
def generator5(shape):
    D=len(shape)
    d=np.full((int(shape[0]/9)+3,9), [1,2,3,2,3,1,3,1,2]).reshape(int(9*(int(shape[0]/9)+3)))
    if D==1:
        data=np.zeros((shape[0]))
        data[0:shape[0]]=d[0:shape[0]]
        answer=np.zeros((1,10))
        answer[:,4]=1
    if D==2:
        data=np.zeros((shape[0],shape[1]))
        for i in range (0,9):
            data[0:shape[0],i::9]=d[i:shape[0]+i].reshape(shape[0],1)
        answer=np.zeros((1,10))
        answer[:,4]=1
    if D==3:
        data=np.zeros((shape[0],shape[1],shape[2]))
        for i in range (0,9):
            for j in range (0,9):
                data[0:shape[0],i::9,j::9]=d[i+j:shape[0]+i+j].reshape(shape[0],1,1)
        answer=np.zeros((1,10))
        answer[:,4]=1
    return data, answer

#same and progression
def generator6(shape):
    D=len(shape)
    d=np.full((int(shape[0]/6)+3,6), [1,2,3,3,3,3]).reshape(int(6*(int(shape[0]/6)+3)))
    d1=np.full((int(shape[0]/3)+3,3), [1,2,3]).reshape(int(3*(int(shape[0]/3)+3)))
    if D==1:
        data=np.zeros((shape[0]))
        data[0:shape[0]]=d[0:shape[0]]
        for i in range (0,shape[0]):
            data[i]+=2*int(i/6)
        answer=np.zeros((1,10))
        answer[:,5]=1
    if D==2:
        data=np.zeros((shape[0],shape[1]))
        data[0:shape[0],:]=d1[0:shape[0]].reshape(shape[0],1)
        answer=np.zeros((1,10))
        answer[:,5]=1
    if D==3:
        data=np.zeros((shape[0],shape[1],shape[2]))
        data[0:shape[0],:,:]=d1[0:shape[0]].reshape(shape[0],1,1)
        answer=np.zeros((1,10))
        answer[:,5]=1
    return data, answer

#progression and same divergence
def generator7(shape):
    D=len(shape)
    #d=np.arange(shape[0])
    #d0=np.full((int(shape[0]/9)+3,9), [1,2,3,2,3,4,3,4,5]).reshape(int(9*(int(shape[0]/9)+3)))
    d1=np.full((int(shape[0]/3)+3,3), [1,2,3]).reshape(int(3*(int(shape[0]/3)+3)))
    if D==1:
        data=np.arange((shape[0]))+1
        for i in range (0,shape[0]):
            data[i]=data[i]-int(i/3)
        #data[0:int(shape[0]/2)]=d[0:shape[0]]
        answer=np.zeros((1,10))
        answer[:,6]=1
    if D==2:
        data=np.zeros((shape[0],shape[1]))
        for i in range (0,shape[1]):
            data[0:shape[0],i]=np.arange((shape[0]))+(i%3)
        answer=np.zeros((1,10))
        answer[:,6]=1
    if D==3:
        data=np.zeros((shape[0],shape[1],shape[2]))
        for i in range (0,shape[1]):
            for j in range (0,shape[2]):
                data[0:shape[0],i,j]=np.arange((shape[0]))+(i%3)+(j%3)
        answer=np.zeros((1,10))
        answer[:,6]=1
    return data, answer

def generator8(shape):
    D=len(shape)
    d=np.full((int(shape[0]/9)+3,9), [1,2,3,2,3,1,3,1,2]).reshape(int(9*(int(shape[0]/9)+3)))
    if D==1:
        data=np.zeros((shape[0]))
        data[0:shape[0]]=d[0:shape[0]]
        for i in range (0,shape[0]):
            data[i]=data[i]+int(i/9)
        
        answer=np.zeros((1,10))
        answer[:,7]=1
    if D==2:
        data=np.zeros((shape[0],shape[1]))
        for i in range (0,shape[1]):
            data[0:shape[0],i]=d[0:shape[0]]+i
        answer=np.zeros((1,10))
        answer[:,7]=1
    if D==3:
        data=np.zeros((shape[0],shape[1],shape[2]))
        for i in range (0,shape[1]):
            for j in range (0,shape[2]):
                data[0:shape[0],i,j]=d[0:shape[0]]+i+j
        answer=np.zeros((1,10))
        answer[:,7]=1
    return data, answer

def generator9(shape):
    D=len(shape)
    if D==1:
        data=np.zeros(shape[0])
        data[0:2]=1
        for i in range (2,shape[0]):
            data[i]=data[i-1]+data[i-2]
        answer=np.zeros((1,10))
        answer[:,8]=1
    if D==2:
        data=np.zeros((shape[0],shape[1]))
        data[0:2,0:2]=1
        for i in range (2,shape[0]):
            data[i,0:2]=data[i-1,0:2]+data[i-2,0:2]
        for j in range (2,shape[1]):
            data[:,j]=data[:,j-1]+data[:,j-2]
        answer=np.zeros((1,10))
        answer[:,8]=1
    if D==3:
        data=np.zeros((shape[0],shape[1],shape[2]))
        data[0:2,0:2,0:2]=1
        for i in range (2,shape[0]):
            data[i,0:2,0:2]=data[i-1,0:2,0:2]+data[i-2,0:2,0:2]
        for j in range (2,shape[1]):
            data[:,j,0:2]=data[:,j-1,0:2]+data[:,j-2,0:2]
        for k in range (2,shape[2]):
            data[:,:,k]=data[:,:,k-1]+data[:,:,k-2]    
        answer=np.zeros((1,10))
        answer[:,8]=1
    return data, answer

def generator10(shape):
    D=len(shape)
    if D==1:
        data=np.arange(shape[0])*2  
        answer=np.zeros((1,10))
        answer[:,9]=1
    if D==2:
        data=np.zeros((shape[0],shape[1]))
        for i in range (0,shape[1]):
            data[0:shape[0],i]=np.arange(shape[0])*2+2*i
        answer=np.zeros((1,10))
        answer[:,9]=1
    if D==3:
        data=np.zeros((shape[0],shape[1],shape[2]))
        for i in range (0,shape[1]):
            for j in range (0,shape[2]):
                data[0:shape[0],i,j]=np.arange(shape[0])*2+2*i+2*j
        answer=np.zeros((1,10))
        answer[:,9]=1
    return data, answer

                       
        