import numpy as np
import tensorflow as tf
# from prepare import prep
# from pic_n import pic3,pic3_2
# from t import gentest
from gen_r import gen_r,gen_r_test,gen_r_ref
def fullnet(fc1, weights, biases, dropout):
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out,fc1
# Training Parameters
def train_fcon(i,ans):
    Num=0
    learning_rate = 0.001
    num_steps = 500
    batch_size = 128
    display_step = 1

    # Network Parameters
    num_input = 13# MNIST data input (img shape: 28*28)
    num_classes = 7  # MNIST total classes (0-9 digits)
    dropout = 0.75 # Dropout, probability to keep units
    # tf Graph input
    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
    weights = {

        'wd1': tf.Variable(tf.random_normal([13, 1024])),
        # 1024 inputs, 10 outputs (class prediction)

        'out': tf.Variable(tf.random_normal([1024, num_classes]))
    }

    biases = {
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
    # Construct model
    logits,fc1 = fullnet(X, weights, biases, keep_prob)
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


    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        #batch_x1, batch_y1 = pic3_2(pic,weight1,weight2)
        #batch_x2, batch_y2 = pic3_2(pic,weight1,weight2)
        test,ans=gen_r_test(i,ans)
        ref,a=gen_r_ref(i)
        if ref[ref!=0].shape[0]>0:
            for step in range(1, num_steps+1):
                batch_x, batch_y = gen_r(i)
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
                    if acc>=0.95:
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
                    #print("Step " + str(step) + ", Minibatch Loss= " + \
                          #"{:.4f}".format(loss) + ", Valication Accuracy= " + \
                          #"{:.3f}".format(acc))
            #print("Optimization Finished!")

            #Calculate accuracy for 256 MNIST test images
            ll=np.zeros((16,1024))
            for i in range (0,16):
                #print("Testing Accuracy:", \
    #                  sess.run(accuracy, feed_dict={X: test[i,:].reshape(1,13),
    #                                                Y: ans[i,:].reshape(1,7),
    #                                                keep_prob: 1.0}))
                l=fc1.eval(feed_dict={X: test[i,:].reshape(1,13),
                                    Y: ans[i,:].reshape(1,7),
                                    keep_prob: 1.0})
                ll[i,:]=l
            lref=np.zeros((7,1024))
            for i in range (0,7):
                l=fc1.eval(feed_dict={X: ref[i,:].reshape(1,13),
                                    Y: a[i,:].reshape(1,7),
                                    keep_prob: 1.0})
                lref[i,:]=l
            lref1=(np.abs(lref[1,:]-lref[0,:])+np.abs(lref[4,:]-lref[3,:]))*0.5
            #lref2=lref[2,:]+lref[5,:]
            la=np.zeros((8,1024))
            EE2=np.zeros(8)
            EE1=np.zeros(8)
            for i in range (0,8):
                la[i,:]=np.abs(ll[i*2,:]-lref[6,:])
                EE1[i]=np.abs(la[i,:]-lref1).sum()
                EE2[i]=np.abs(ll[i*2+1,:]-0.5*(lref[2,:]+lref[5,:])).sum()
            EE=EE1*2/3+EE2/3
        else:
            EE1=np.zeros(8)
            EE2=np.zeros(8)
            for i in range (0,8):
                EE1[i]=np.abs(test[i*2,:]).sum()
                EE2[i]=np.abs(test[i*2+1,:]).sum()
                EE=EE1*2/3+EE2/3
            Num=0
    return EE1,EE2,EE,Num