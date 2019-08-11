from foward_error_ex2 import foward_error
import random
for i in range (0,5):
    k=random.shuffle([0,3,4,5,6,7])
    a=random.randint(len(k))
    foward_error(1,k[0:a],5,'1_c_ex_f')

