import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import random
import math
import re
import time
import numpy as np
def ans_cut(N):
    a=Image.open('/Users/sofiadunlosky/Downloads/ilovepdf_images-extracted/img'+N+'.jpg')
    pic=np.zeros((50,60,6))
    pic[:,:,0]=np.array(a)[215:265,15:75,0]
    pic[:,:,3]=np.array(a)[293:343,15:75,0]
    pic[:,:,1]=np.array(a)[215:265,125:185,0]
    pic[:,:,4]=np.array(a)[293:343,125:185,0]
    pic[:,:,2]=np.array(a)[215:265,235:295,0]
    pic[:,:,5]=np.array(a)[293:343,235:295,0]
    p=np.array(a)[100:150,110:170,0]
    for i in range (0,6):
        plt.subplot(2,3,i+1)
        plt.imshow(pic[:,:,i])
    return p,pic,a