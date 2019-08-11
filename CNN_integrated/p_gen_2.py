from PIL import Image
import numpy as np
import random
def p_gn(N):
    if N=='11':
        a=Image.open('/afs/inf.ed.ac.uk/user/s18/s1883226/Downloads/experiment/ilovepdf_images-extracted/img'+N+'.png')
    else:
        a=Image.open('/afs/inf.ed.ac.uk/user/s18/s1883226/Downloads/experiment/ilovepdf_images-extracted/img'+N+'.jpg')
    if random.randint(0,3)<2:
        n=random.randint(30,100)
        m=random.randint(65,110)
        p=np.array(a)[n:n+50,m:m+60]
    else:
        n=random.randint(30,40)
        m=random.randint(175,205)
        p=np.array(a)[n:n+50,m:m+60]
    return p
