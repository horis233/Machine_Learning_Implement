import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def run(model,checkpoint):

    # Build computation graph
    W = {
        'layer_1': tf.Variable(tf.random_normal([n_input, n_hidden_1], mean=0.0, stddev=3.0 / (n_input + n_hidden_1))),
        'output': tf.Variable(
            tf.random_normal([n_hidden_1, n_classes], mean=0.0, stddev=3.0 / (n_hidden_1 + n_classes)))
    }

    print("load multi class data")

    # Initialize session
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    if model == "full":
        print("get checkpoint "+ str(checkpoint)+ " from " + model)
        saver.restore(sess,"/home/horis/Documents/Machine Learning examples/Neural Networks/Feedforward_models/model"+str(checkpoint)+".ckpt")
    if model == "drop":
        print("get checkpoint " + str(checkpoint) + " from " + model)
        saver.restore(sess,"/home/horis/Documents/Machine Learning examples/Neural Networks/Dropout_models/model" + str(checkpoint) + ".ckpt")
    W = sess.run(W)
    visualization_W = W['layer_1'].T
    visualization_reshape = visualization_W.reshape(1000, 28, 28)

    return visualization_reshape


def print_graph(model,checkpoint):
    w = run(model,checkpoint)
    plt.figure()
    for i in range(1000):
        #plt.subplot(32,32,i+1)
        plt.imshow(w[i], cmap=plt.cm.gray)
        '''
        if model == "full":
             plt.savefig('/home/horis/Documents/Machine Learning examples/Neural Networks/Feedforward_pic/'+str(i)+'weight of chechpoint: '+ str(checkpoint)+'.png' )
        if model == "drop":
             plt.savefig('/home/horis/Documents/Machine Learning examples/Neural Networks/Dropout_pic/'+str(i)+'weight of chechpoint: '+ str(checkpoint)+'.png' )
        '''
    if model == "full":
        plt.savefig('/home/horis/Documents/Machine Learning examples/Neural Networks/Feedforward_pic/weight of '
                    'chechpoint: ' + str(checkpoint) + '.png')
    if model == "drop":
        plt.savefig('/home/horis/Documents/Machine Learning examples/Neural Networks/Dropout_pic/weight of chechpoint: '
                    + str(checkpoint) + '.png')
    plt.show()


if __name__ == '__main__':

    # Parameters
    n_hidden_1 = 1000  # 1st layer number of features
    n_input = 784  # MNIST data input (img shape: 28*28)
    n_classes = 10  # MNIST total classes (0-9 digits)
    checkpoint = [5,10,15,20]
    model = ["full","drop"]
    # Change index from model[0] to model[1] and from checkpoint[0] to checkpoint[1]
    print_graph(model[0], checkpoint[3])
    print('finish')