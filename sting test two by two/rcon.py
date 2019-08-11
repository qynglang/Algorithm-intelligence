import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from prepare import prep
from data_gene import forward
#from pic_n import pic3,pic3_2
#from t import gentest
from r_gene import g_error
def RNN(x, weights, biases):
    timesteps = 2
    num_hidden = 128
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    #print(lstm_cell.eval(sess))
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out'],outputs[-1]
# Training Parameters
def train_rcon(pic,weight1,weight2,n,bs,w):
    tf.reset_default_graph()
    learning_rate = 0.001
#     training_steps = 10000
#     batch_size = 128
#     display_step = 200

#     # Network Parameters
#     num_input = 28 # MNIST data input (img shape: 28*28)
#     timesteps = 28 # timesteps
#     num_hidden = 128 # hidden layer num of features
#     num_classes = 10 # MNIST total classes (0-9 digits)

#     # tf Graph input
#     X = tf.placeholder("float", [None, timesteps, num_input])
#     Y = tf.placeholder("float", [None, num_classes])
    Num=0
    learning_rate = 0.001
    num_steps = n
    batch_size = bs
    display_step = 1

    # Network Parameters
    num_input = 180 # MNIST data input (img shape: 28*28)
    num_classes = 2 # MNIST total classes (0-9 digits)
    timesteps = 2
    num_hidden = 128
    dropout = 0.75 # Dropout, probability to keep units
    # tf Graph input
    X = tf.placeholder(tf.float32, [None, timesteps, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
    logits, outputs = RNN(X, weights, biases)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    
    init = tf.global_variables_initializer()
#     # Construct model
#     logits = fullnet(X, weights, biases, keep_prob)
#     prediction = tf.nn.softmax(logits)

#     # Define loss and optimizer
#     loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#         logits=logits, labels=Y))
#     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#     train_op = optimizer.minimize(loss_op)


#     # Evaluate model
#     correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#     # Initialize the variables (i.e. assign their default value)
#     init = tf.global_variables_initializer()


    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        #batch_x1, batch_y1 = pic3_2(pic,weight1,weight2)
        #batch_x2, batch_y2 = pic3_2(pic,weight1,weight2)
        #test,ans=gentest(pic,weight1,weight2,ans,a,bs)
        for step in range(1, num_steps+1):
            batch_x, batch_y = g_error(pic,weight1,weight2)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y,
                                                                     keep_prob: 1.0})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Train Accuracy= " + \
                      "{:.3f}".format(acc))
                if acc>=0.8:
                    if step>5:
                        Num=1
                        break
                 
                
    #with tf.variable_scope("model", reuse=True):
        tr=1
        EE=np.zeros((6,tr))
        data=np.load('data_4.npy')
    #EE2=np.zeros((6,tr))
        for j in range (0,tr):
            II=forward(data[:,:,:,w],weight1,weight2)
            data=np.zeros((1,8,180))
            data[:,0,:]=II[0,:].reshape(1,180)
            data[:,1,:]=II[1,:].reshape(1,180)
            data[:,2,:]=II[2,:].reshape(1,180)
            data[:,4,:]=II[0,:].reshape(1,180)
            data[:,5,:]=II[2,:].reshape(1,180)
            data[:,6,:]=II[1,:].reshape(1,180)
            sess.run(accuracy, feed_dict={X: data[:,0:2,:], Y: np.zeros((1,2))})
            l=outputs.eval(feed_dict={X: data[:,0:2,:],
                            Y: np.zeros((1,2)),
                            keep_prob: 1.0})
            l1=outputs.eval(feed_dict={X: data[:,4:6,:],
                            Y: np.zeros((1,2)),
                            keep_prob: 1.0})
            for i in range (1,7):
                III=II[2+i,:].reshape(1,180)
                data[:,3,:]=III.reshape(1,180)
                data[:,7,:]=III.reshape(1,180)
                l3=outputs.eval(feed_dict={X: data[:,2:4,:],
                            Y: np.zeros((1,2)),
                            keep_prob: 1.0})
                l4=outputs.eval(feed_dict={X: data[:,6:8,:],
                            Y: np.zeros((1,2)),
                            keep_prob: 1.0})
#                 O1=RNN(data[:,0:2,:],weights, biases)
#                 O2=RNN(data[:,2:4,:],weights, biases)
#                 O3=RNN(data[:,4:6,:],weights, biases)
#                 O4=RNN(data[:,6:8,:],weights, biases)
                EE[i-1,j]=np.abs(l3-l).sum()+np.abs(l4-l1).sum()
            N=np.zeros(tr)
            for i in range (0,tr):
                N[i]=np.argmin(EE[:,i]) 
        sess.close()

                #loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x1,
                                                                     #Y: batch_y1,
                                                                     #keep_prob: 1.0})
                #loss1, acc1 = sess.run([loss_op, accuracy], feed_dict={X: batch_x2,
                                                                     #Y: batch_y2,
                                                                     #keep_prob: 1.0})
                #loss=(loss+loss1)/2
                #acc=(acc+acc1)/2
                #print("Step " + str(step) + ", Minibatch Loss= " + \
                      #"{:.4f}".format(loss) + ", Valication Accuracy= " + \
                      #"{:.3f}".format(acc))
        #print("Optimization Finished!")

        #Calculate accuracy for 256 MNIST test images
        #print("Testing Accuracy:", \
             #sess.run(accuracy, feed_dict={X: test,
                                           #Y: ans,
                                           #keep_prob: 1.0}))
#         p=prediction.eval(feed_dict={X: test,
#                             Y: ans,
#                             keep_prob: 1.0})
#         l=logits.eval(feed_dict={X: test,
#                             Y: ans,
#                             keep_prob: 1.0})
        #w=weights['out'].eval(sess)
        #b=biases['out'].eval(sess)
        
    return EE,N,Num