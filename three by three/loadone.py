import numpy as np
import random
from rotate import get_img_rot_broa as rotate
def load_sample(Pic):
    data0=np.zeros((8,784))
    answer=np.zeros((8,2))
    for i in range (0,8):
        if random.randint(0,2)==1:
            n1=random.randint(0,10)
            n2=random.randint(0,10)
            pic1=np.ones((28,28))*255
            pic1[n1:27,n2:27]=Pic[0:27-n1,0:27-n2]
            #pic1=misc.imresize(np.array(Pic)[33+n1:73+n1,79+n2:121+n2,:], (28, 28))[:,:,0]
            m=random.randint(0,20)
            #pic1[pic1<=190+m]=0
            #pic1[pic1>190+m]=1
            #pic1=pic1/255
            alpha=random.randint(0,360)
            #n3=random.randint(0,4)
            #n4=random.randint(0,4)
            if random.randint(0,2)==1:
                pic2=rotate(np.uint8(pic1), degree=alpha, filled_color=[255,255,255])
            else:
                pic2=rotate(np.uint8(pic1[::-1]), degree=alpha, filled_color=[255,255,255])
            data0[i,:]=pic2[pic2.shape[0]-28:pic2.shape[0],pic2.shape[1]-28:pic2.shape[1]].reshape(1,784)/255
            #if i==0:
                #plt.imshow(data0[i,:].reshape(28,28))
            #data[i,:]=pic2[4+n3:32+n3,4+n4:32+n4].reshape(1,784)
            answer[i,:]=[1,0]
        else:
            data0[i,:]=np.random.randint(255, size=(1,784))/255
            answer[i,:]=[0,1]
    return data0,answer