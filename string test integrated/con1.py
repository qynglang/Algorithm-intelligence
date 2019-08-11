import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#from rotate import get_img_rot_broa as rotate
#from loadone1 import load_sample as load
from smg import load,load_test
from convo1 import conv2d,maxpool2d,conv_net
def train_cov(m):
    # Training Parameters
    Num=0
    learning_rate = 0.0001
    num_steps = 3000
    batch_size = 8
    display_step = 100

    # Network Parameters
    num_input = 3000 # MNIST data input (img shape: 28*28)
    num_classes = m # MNIST total classes (0-9 digits)
    dropout = 0.75 # Dropout, probability to keep units

    # tf Graph input
    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([10, 10, 1, 3])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([25, 25, 3, 6])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([288, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, num_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([3])),
        'bc2': tf.Variable(tf.random_normal([6])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # Construct model
    logits = conv_net(X, weights, biases, keep_prob)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)


    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # weight1=np.zeros((5,5,1,3))
    # weight2=np.zeros((5,5,3,6))
    # acc1=0
    # Start training
#     test=np.zeros((16,784))
#     answer_test=np.zeros((16,2))
#     for k in range(0,16):
#         test[k,:]=A.reshape(784,)/255
#         answer_test[k,:]=[1,0]
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        #batch_x1, batch_y1 = load(A)
        #batch_x2, batch_y2 = load(A)
        #a=np.zeros(num_steps+1)
        ac_max=0
        for step in range(1, num_steps+1):
            #batch_x, batch_y = load(A)
            batch_x, batch_y = load([50,60],8,m)
            v_x, v_y = load([50,60],8,m)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x.reshape(8,3000), Y: batch_y[:,0:m], keep_prob: dropout})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x.reshape(8,3000),
                                                                     Y: batch_y[:,0:m],
                                                                     keep_prob: 1.0})
                los, ac = sess.run([loss_op, accuracy], feed_dict={X: v_x.reshape(8,3000),
                                                                     Y: v_y[:,0:m],
                                                                     keep_prob: 1.0})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Train Accuracy= " + \
                      "{:.3f}".format(acc)+",Validation Accuracy= " +\
                     "{:.3f}".format(ac))
                #a[step]=ac
                if ac_max<ac:
                    save_path = saver.save(sess, 'model/model1'+str(m)+'.ckpt')
                    ac_max=ac
                if ac==1:
                    if step>5:
                        Num=1
                        break
                    
                #loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x1,
                                                                     #Y: batch_y1,
                                                                     #keep_prob: 1.0})
                #loss1, acc1 = sess.run([loss_op, accuracy], feed_dict={X: batch_x2,
                                                                      #Y: batch_y2,
                                                                      #keep_prob: 1.0})
                #loss=(loss+loss1)/2
                #acc=(acc+acc1)/2
    #             if acc>acc1:
    #                 print('1')
    #                 weight1=weights['wc1'].eval(sess)
    #                 weight2=weights['wc2'].eval(sess)
    #                 acc1=acc
                #print("Step " + str(step) + ", Minibatch Loss= " + \
                      #"{:.4f}".format(loss) + ", Valication Accuracy= " + \
                      #"{:.3f}".format(acc))
        #print("Optimization Finished!")

        #Calculate accuracy for 256 MNIST test images
        #step=np.argmax(a)
        saver.restore(sess, 'model/model1'+str(m)+'.ckpt')
        x, y = load_test([50,60],100,m)
        acc=np.zeros(m)
        for i in range (0,m):
            acc[i]=sess.run(accuracy, feed_dict={X: x[i*10:i*10+8,:].reshape(8,3000),
                                           Y: y[i*10:i*10+8,0:m].reshape(8,m),
                                           keep_prob: 1.0})
#         weight1=weights['wc1'].eval(sess)
#         weight2=weights['wc2'].eval(sess)
#     return weight1,weight2,Num
    return acc




