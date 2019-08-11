from rcon import train_rcon
import numpy as np
def forward(N,ans):
    tr=50
    EEE=np.zeros((6,tr))
    NN=np.zeros(tr)
    ac=np.zeros(tr)
    for i in range (0,tr):
        EE,NNN,Num,acc=train_rcon(200,N,ans)
        EEE[:,i]=EE
        NN[i]=NNN
        ac[i]=acc
    print(np.argsort(np.average(EEE,axis=1)))
    print(np.where(NN==ans)[0].size)
    print(np.average(EEE,axis=1)[ans]/EEE.mean())
    print(ac.mean())
    np.save('Data/'+N+str('EEE_rnn_1')+'.npy',EEE)
    np.save('Data/'+N+str('NN_rnn_1')+'.npy',NN)
    np.save('Data/'+N+str('ac_rnn_1')+'.npy',ac)