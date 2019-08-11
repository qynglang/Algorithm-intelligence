import numpy as np
from prepare import prep
def gentest(pic,weight1,weight2,ans,a,bs):
    data=np.zeros((bs,49*2*a*3*weight1.shape[4]))
    answer=np.zeros((bs,2))
    for i in range (0,8):
        num=[6,7,i+8]
        data[i,:]=prep(pic[:,:,num],weight1,weight2,a).reshape(1,7*7*2*a*3*weight1.shape[4])
        if i==ans:
            answer[i,:]=[1,0]
        else:
            answer[i,:]=[0,1]
    if bs>8:
        for j in range (0,int(bs/8)):
            data[8*(j+1):8*(j+1)+8,:]=data[0:8,:]
            answer[8*(j+1):8*(j+1)+8,:]=answer[0:8,:]
    return data, answer