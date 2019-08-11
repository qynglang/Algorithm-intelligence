from forward_error_weight_f import foward_error
import random
for i in range (0,5):
    k=[0,1,2,3,4,5,7]
    random.shuffle(k)
    a=random.randint(2,len(k)-1)
    foward_error(3,k[0:a],1,'3_c_f'+str(i))