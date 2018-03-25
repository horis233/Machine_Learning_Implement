import tensorflow as tf
import numpy as np
import numpy
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score


def loaddata():
    # Loading my data

    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx] / 255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
        trainData_rs = trainData.reshape(trainData.shape[0], -1)
        validData_rs = validData.reshape(validData.shape[0], -1)
        testData_rs = testData.reshape(testData.shape[0], -1)
    return trainData_rs, trainTarget, validData_rs, validTarget, testData_rs, testTarget


def Training(l_rate, Lambda):
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[784, 1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32, [None, 1], name='target_y')

    # Graph definition
    y_predicted = tf.matmul(X, W) + b

    # Error definition
    squared_error = tf.square(y_predicted - y_target)
    loss = tf.reduce_mean(squared_error, name='mean_squared_error')/2 + tf.reduce_sum(tf.square(W)) * Lambda / 2

    # Training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate).minimize(loss)

    return X, y_target, y_predicted, loss, optimizer


def run_ne(Lam):

    starttime = time.time()
    trainData, trainTarget, validData, validTarget, testData, testTarget = loaddata()
    m = trainData.shape[0]
    valid_length = validData.shape[0]
    test_length = testData.shape[0]
    traindata_with_bias = np.c_[np.ones((m,1)),trainData]
    validdata_with_bias = np.c_[np.ones((valid_length,1)),validData]
    testdata_with_bias = np.c_[np.ones((test_length,1)),testData]
    X = tf.constant(traindata_with_bias,dtype=tf.float32,name='input_x')
    x_valid = tf.constant(validdata_with_bias,dtype=tf.float32,name='valid_x')
    x_test = tf.constant(testdata_with_bias,dtype=tf.float32,name='test_x')
    y_target = tf.constant(trainTarget,dtype=tf.float32,name='target_y')
    y_valid = tf.constant(validTarget,dtype=tf.float32,name='valid_target')
    y_test = tf.constant(testTarget,dtype=tf.float32,name='test_target')
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    # train
    X_T = tf.transpose(X)
    W = tf.matmul(tf.matmul(tf.matrix_inverse( tf.matmul(X_T,X)),X_T),y_target)
    duration = time.time() - starttime


    # validation loss
    y_predicted_valid = tf.matmul(x_valid, W)
    squared_error_valid = tf.reduce_mean(tf.square(y_predicted_valid - validTarget), reduction_indices=1, name='squared_error')
    loss_valid = tf.reduce_mean(squared_error_valid, name='mean_squared_error') + tf.reduce_sum(tf.square(W)) * Lam / 2
    loss_v = sess.run(loss_valid)

    # validation accuracy
    y_p = np.round(sess.run(y_predicted_valid))
    accuracy_valid = accuracy_score(y_p, y_valid.eval())

    # print validation loss and accuracy
    print("valid MSE: ", loss_v)
    print("valid accuracy: ", accuracy_valid)

    # test loss
    y_predicted_test = tf.matmul(x_test, W)
    squared_error_test = tf.reduce_mean(tf.square(y_predicted_test - testTarget), reduction_indices=1, name='squared_error')
    loss_test = tf.reduce_mean(squared_error_test, name='mean_squared_error') + tf.reduce_sum(tf.square(W)) * Lam / 2
    loss_t = sess.run(loss_test)

    # test accuracy
    y_p = np.round(sess.run(y_predicted_test))
    accuracy_test = accuracy_score(y_p, y_test.eval())

    # print validation loss and accuracy
    print("test MSE: ", loss_t)
    print("test accuracy: ", accuracy_test)

    # duration
    print("Duration :", duration)

def run(lr, lam, batch_size, num_epoch):
    startTime = time.time()
    # Build computation graph
    X, y_target, y_predicted, mse, train = Training(lr, lam)

    # Loading my data
    trainData, trainTarget, validData, validTarget, testData, testTarget = loaddata()

    # Initialize session
    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init)

    loss_list = []
    accuracy_list = []
    num_update = 0

    rnd_idx = np.arange(trainData.shape[0])
    num_train_cases = trainData.shape[0]
    if batch_size == -1:
        batch_size = num_train_cases
    num_steps = int(np.ceil(num_train_cases / batch_size))
    while num_update < num_epoch:
        np.random.shuffle(rnd_idx)
        inputs_train = trainData[rnd_idx]
        target_train = trainTarget[rnd_idx]

        for step in range(num_steps):
            # Select random minibatch
            # indices = np.random.choice(trainData.shape[0], batch_size)
            # X_batch, y_batch = trainData[indices], trainTarget[indices]

            # Select minibatch
            start = step * batch_size
            end = min(num_train_cases, (step + 1) * batch_size)
            x = inputs_train[start: end]
            t = target_train[start: end]

            _, err, y_prediction = sess.run([train, mse, y_predicted], feed_dict={X: x, y_target: t})

            training_accuracy = np.mean(y_target == y_prediction)

            # wList.append(currentW)
            # loss_list.append(err)

            if num_update < num_epoch:
                num_update += 1
            else:
                break
        duration = time.time() - startTime
        print('Epoch: {:4}, Loss: {:5f}, Accuracy:{:5f},Duration: {:2f}'.format(int(num_update / num_steps), err, training_accuracy, duration))
        loss_list.append(err)
        accuracy_list.append(training_accuracy)

    duration = time.time() - startTime

    # Test on the validation setsess.run(result1,feed_dict={x: [1,2,1]})
    valid_err, valid_predict = sess.run([mse, y_predicted], feed_dict={X: validData, y_target: validTarget})
    valid_predict = np.round(valid_predict)
    # Testing model
    test_err, test_predict = sess.run([mse, y_predicted], feed_dict={X: testData, y_target: testTarget})
    test_predict = np.round(test_predict)
    # accuracy
    valid_accuracy = accuracy_score(validTarget, valid_predict)
    test_accuracy = accuracy_score(testTarget, test_predict)

    print("valid MSE: %.2f" % valid_err)
    print("Valid Accuracy: %.2f" % valid_accuracy)
    print("test MSE: %.2f" % test_err)
    print("Test Accuracy: %.2f" %test_accuracy)

    print("Duration: %.2f" %duration)

    return loss_list


def print_graph(question_number):
    if question_number == '1.1':
        loss_lr1 = run(lr=0.005, lam=0, batch_size=500, num_epoch=20000)
        loss_lr2 = run(lr=0.001, lam=0, batch_size=500, num_epoch=20000)
        loss_lr3 = run(lr=0.0001, lam=0, batch_size=500, num_epoch=20000)
        plt.title('Loss of Linear Regression for Different Learning Rates')
        lr1, = plt.plot(loss_lr1, label="Learning Rate: 0.005", color='r')
        lr2, = plt.plot(loss_lr2, label="Learning Rate: 0.001", color='b')
        lr3, = plt.plot(loss_lr3, label="Learning Rate: 0.0001", color='g')
        plt.legend(handles=[lr1, lr2, lr3], loc='upper right')
        plt.ylabel('loss')
        plt.xlabel('num of epoch')
        plt.show()
    if question_number == '1.2':
        loss_bs1 = run(lr=0.005, lam=0, batch_size=500, num_epoch=20000)
        loss_bs2 = run(lr=0.005, lam=0, batch_size=1500, num_epoch=20000)
        loss_bs3 = run(lr=0.005, lam=0, batch_size=3500, num_epoch=20000)
        plt.title('Loss of Linear Regression for Different Batch Size')
        bs1, = plt.plot(loss_bs1, label="Batch size: 500", color='r')
        bs2, = plt.plot(loss_bs2, label="Batch size: 1500", color='b')
        bs3, = plt.plot(loss_bs3, label="Batch size: 3500", color='g')
        plt.legend(handles=[bs1, bs2, bs3], loc='upper right')
        plt.ylabel('loss')
        plt.xlabel('num of epoch')
        plt.show()
    if question_number == '1.3':
        loss_lam1 = run(lr=0.005, lam=0, batch_size=500, num_epoch=20000)
        loss_lam2 = run(lr=0.005, lam=0.001, batch_size=500, num_epoch=20000)
        loss_lam3 = run(lr=0.005, lam=0.1, batch_size=500, num_epoch=20000)
        loss_lam4 = run(lr=0.005, lam=1, batch_size=500, num_epoch=20000)
        plt.title('Loss of Linear Regression for Different Lambda')
        lam1, = plt.plot(loss_lam1, label="lambda: 0", color='r')
        lam2, = plt.plot(loss_lam2, label="lambda: 0.001", color='b')
        lam3, = plt.plot(loss_lam3, label="lambda: 0.1", color='g')
        lam4, = plt.plot(loss_lam4, label="lambda: 1", color='y')
        plt.legend(handles=[lam1, lam2, lam3, lam4], loc='upper right')
        plt.ylabel('loss')
        plt.xlabel('num of epoch')
        plt.show()
    if question_number == '1.4':
        run_ne(0.1)



if __name__ == '__main__':

    print_graph('1.1')
    # print_graph('1.2')
    # print_graph('1.3')
    # print_graph('1.4')

    print('finish')