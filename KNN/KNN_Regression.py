import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Q1


'''
Data Loading
'''
np.random.seed(521)
Data = np.linspace(1.0,10.0,num=100)[:,np.newaxis]
Target = np.sin(Data) + 0.1 * np.power(Data,2) + 0.5 * np.random.randn(100,1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData,trainTarget = Data[randIdx[:80]],Target[randIdx[:80]]
validData,validTarget = Data[randIdx[80:90]],Target[randIdx[80:90]]
testData,testTarget = Data[randIdx[90:100]],Target[randIdx[90:100]]


def find_knn(pairwise_distance, k_n):
    """

    :param pairwise_distance: the Euclidean distance between training data and test data
    :param k_n: the number of nearest neighbours.
    :return: responsibility
    """
    # n is the number of training number
    n = tf.shape(pairwise_distance)[1]
    m = tf.shape(pairwise_distance)[0]
    # Changing distance into negative distance in order to get top k.
    neg_distance = - pairwise_distance
    # indices: The indices of values within the last dimension of input.
    _,indices = tf.nn.top_k(neg_distance, k=k_n, sorted=False)
    in_exp = tf.expand_dims(indices,2)
    r = tf.to_float(tf.equal(in_exp,tf.reshape(tf.range(n),[1,1,-1])))
    r_sum = tf.reduce_sum(r,1)
    r_aver = r_sum / tf.to_float(k_n)
    return r_aver


def prediction(r, training_target):
    """

    :param r: responsibility
    :param training_target: target of training
    :return: prediction target
    """
    return tf.matmul(r, training_target)


def mse(test, prediction):
    """

    :param test: true target
    :param prediction: prediction target
    :return: mean square error
    """
    sqr_err = tf.square(test - prediction)
    loss = tf.reduce_mean(tf.reduce_sum(sqr_err, 1))/2
    return loss


# initialize inputs and targets
trainingset_x = tf.placeholder(tf.float32)
trainingset_y = tf.placeholder(tf.float32)
test_x = tf.placeholder(tf.float32)
test_y = tf.placeholder(tf.float32)
k = tf.placeholder("int32")

# calculating Euc_distance
dis = Q1.distanceFunc(test_x,trainingset_x)
# picking KNN and responsibility
res = find_knn(dis, k_n=k)
# calculating the prediction
prediction_y = prediction(res,trainingset_y)
# mean squared error
MSE = mse(test_y, prediction_y)

# interact
sess = tf.InteractiveSession()

X = np.linspace(0.0,11.0,num=1000)[:,np.newaxis]
# Find the nearest k neighbours:
ks = [1,3,5,50]
min_valid = float("inf")

for kc in ks:
    # train mse
    feed_dict_train = {trainingset_x: trainData, trainingset_y: trainTarget, test_x: trainData, test_y: trainTarget, k: kc}
    mse_train = sess.run(MSE, feed_dict_train)
    # validation mse
    feed_dict_valid = {trainingset_x: trainData, trainingset_y: trainTarget, test_x: validData, test_y: validTarget, k: kc}
    mse_valid = sess.run(MSE, feed_dict_valid)
    # test mse
    feed_dict_test = {trainingset_x: trainData, trainingset_y: trainTarget, test_x: testData, test_y: testTarget, k: kc}
    mse_test = sess.run(MSE, feed_dict_test)
    print("training MSE = {}, validation MSE = {}, test MSE = {} with hyper-parameter {}\n" .format(mse_train,mse_valid,mse_test,kc))
    feed_dict_prediction = {trainingset_x: trainData, trainingset_y: trainTarget, test_x: X, k: kc}
    prediction = sess.run(prediction_y,feed_dict_prediction)
    plt.figure(kc)
    plt.plot(trainData,trainTarget,'.b')
    plt.title("K-NN regression on data1D, k={}".format(kc))
    plt.plot(X, prediction, '-g')
    plt.show()
    if mse_valid < min_valid:
        min_valid = mse_valid
        k_best = kc
print("best K using validation set is K={}\n".format(k_best))
