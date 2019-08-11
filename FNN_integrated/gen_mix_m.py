import numpy as np
def mix_m(n,pic):
    b=pic[:,:,n].copy()
    pic1=np.zeros((50,60,6))
    for i in range (0,6):
        if i!=n:
            b=pic[:,:,n].copy()
            b[5:45,5:55]=pic[:,:,i][5:45,5:55]
            pic1[:,:,i]=b
    pic1[:,:,n]=pic[:,:,n]
    return pic1
def mix_m1(n,pic):
    b=pic[:,:,n].copy()
    pic1=np.zeros((50,60,6))
    for i in range (0,6):
        if i!=n:
            b=pic[:,:,n].copy()
            b[25::,:]=pic[:,:,i][25::,:]
            pic1[:,:,i]=b
    pic1[:,:,n]=pic[:,:,n]
    return pic1
def mix_m2(n,pic):
    b=pic[:,:,n].copy()
    pic1=np.zeros((50,60,6))
    for i in range (0,6):
        if i!=n:
            b=pic[:,:,n].copy()
            b[:,30::]=pic[:,:,i][:,30::]
            pic1[:,:,i]=b
    pic1[:,:,n]=pic[:,:,n]
    return pic1
def mix_m3(n,pic):
    b=pic[:,:,n].copy()
    pic1=np.zeros((50,60,6))
    for i in range (0,6):
        if i!=n:
            b=pic[:,:,n].copy()
            b[:,0:30]=pic[:,:,i][:,0:30]
            pic1[:,:,i]=b
    pic1[:,:,n]=pic[:,:,n]
    return pic1
def mix_m4(n,pic):
    b=pic[:,:,n].copy()
    pic1=np.zeros((50,60,6))
    for i in range (0,6):
        if i!=n:
            b=pic[:,:,n].copy()
            b[0:25,:]=pic[:,:,i][0:25,:]
            pic1[:,:,i]=b
    pic1[:,:,n]=pic[:,:,n]
    return pic1