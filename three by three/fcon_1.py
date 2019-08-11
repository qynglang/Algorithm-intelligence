import numpy as np
import tensorflow as tf
from prepare import prep
#from pic_n import pic3,pic3_2
#from t import gentest
from data_gene import g_error
def fullnet(fc1, weights, biases, dropout):
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
# Training Parameters
def train_fcon(pic,weight1,weight2,n,bs):
    Num=0
    learning_rate = 0.001
    num_steps = n
    batch_size = bs
    display_step = 1

    # Network Parameters
    num_input = 7*7*2*3 # MNIST data input (img shape: 28*28)
    num_classes = 4 # MNIST total classes (0-9 digits)
    dropout = 0.75 # Dropout, probability to keep units
    # tf Graph input
    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
    weights = {

        'wd1': tf.Variable(tf.random_normal([7*7*2*3, 1024])),
        # 1024 inputs, 10 outputs (class prediction)

        'out': tf.Variable(tf.random_normal([1024, num_classes]))
    }

    biases = {
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
    # Construct model
    logits = fullnet(X, weights, biases, keep_prob)
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
        w=weights['wd1'].eval(sess)
        b=biases['bd1'].eval(sess)
    return w,b,Num