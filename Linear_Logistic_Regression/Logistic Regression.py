import tensorflow as tf
import numpy as np
import numpy
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def sign(X):
    for i, num in enumerate(X):
        if num >= 0.5:
            X[i] = 1
        else:
            X[i] = 0
    return X

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


def loaddata_multi():
    # Loading my data

    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
        trainData_rs = trainData.reshape(trainData.shape[0], -1)
        validData_rs = validData.reshape(validData.shape[0], -1)
        testData_rs = testData.reshape(testData.shape[0], -1)
        trainTarget1 = (np.arange(10) == np.array(trainTarget)[:, None]).astype(np.float32)
        validTarget1 = (np.arange(10) == np.array(validTarget)[:, None]).astype(np.float32)
        testTarget1 = (np.arange(10) == np.array(testTarget)[:, None]).astype(np.float32)

    return trainData_rs, trainTarget1, validData_rs, validTarget1, testData_rs, testTarget1


def loaddata_face():
    data = np.load('./data.npy') / 255
    data = np.reshape(data, [-1, 32 * 32])
    target = np.load('./target.npy')
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    trBatch = int(0.8 * len(rnd_idx))
    validBatch = int(0.1 * len(rnd_idx))
    trainData, validData, testData = data[rnd_idx[1:trBatch], :], \
                                     data[rnd_idx[trBatch + 1:trBatch + validBatch], :], \
                                     data[rnd_idx[trBatch + validBatch + 1:-1], :]
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], 0], \
                                           target[rnd_idx[trBatch + 1:trBatch + validBatch], 0], \
                                           target[rnd_idx[trBatch + validBatch + 1:-1], 0]

    # reshape input data to be a training point by 1024 dimensions
    trainData_rs = trainData.reshape(trainData.shape[0], -1)
    validData_rs = validData.reshape(validData.shape[0], -1)
    testData_rs = testData.reshape(testData.shape[0], -1)
    trainTarget1 = (np.arange(6) == np.array(trainTarget)[:, None]).astype(np.float32)
    validTarget1 = (np.arange(6) == np.array(validTarget)[:, None]).astype(np.float32)
    testTarget1 = (np.arange(6) == np.array(testTarget)[:, None]).astype(np.float32)

    return trainData_rs, trainTarget1, validData_rs, validTarget1, testData_rs, testTarget1


def Training(l_rate, Lambda, model='SGD'):
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[784, 1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32, [None, 1], name='target_y')

    # Graph definition
    y_predicted = tf.matmul(X, W) + b

    # Error definition
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predicted, labels=y_target)
                          , name='mean_squared_error') + tf.reduce_sum(tf.square(W)) * Lambda / 2

    # Training mechanism
    if model == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate).minimize(loss)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(loss)
    return X, y_target, y_predicted, loss, optimizer


def Training_lr(l_rate, Lambda):
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[784, 1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32, [None, 1], name='target_y')

    # Graph definition
    y_predicted = tf.matmul(X, W) + b

    # Error definition
    squared_error = tf.reduce_mean(tf.square(y_predicted - y_target), reduction_indices=1, name='squared_error')
    loss = tf.reduce_mean(squared_error, name='mean_squared_error') + tf.reduce_sum(tf.square(W)) * Lambda / 2

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(loss)

    return X, y_target, y_predicted, loss, optimizer


def Training_multi(l_rate, Lambda):
    W = tf.Variable(tf.truncated_normal(shape=[784, 10], stddev=0.5), name='weights')
    b = tf.Variable(tf.zeros([10]), name='biases')
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32, [None, 10], name='target_y')

    # Graph definition
    y_predicted = tf.matmul(X, W) + b

    # Error definition
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y_predicted)
                          , name='mean_squared_error') + tf.reduce_sum(tf.square(W)) * Lambda / 2

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(loss)
    return X, y_target, y_predicted, loss, optimizer


def Training_multi_face(l_rate, Lambda):
    print("training face")
    W = tf.Variable(tf.truncated_normal(shape=[1024, 6], stddev=0.5), name='weights')
    b = tf.Variable(tf.zeros([6]), name='biases')
    X = tf.placeholder(tf.float32, [None, 1024], name='input_x')
    y_target = tf.placeholder(tf.float32, [None, 6], name='target_y')

    # Graph definition
    y_predicted = tf.matmul(X, W) + b

    # Error definition
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y_predicted)
                          , name='mean_squared_error') + tf.reduce_sum(tf.square(W)) * Lambda / 2

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(loss)
    return X, y_target, y_predicted, loss, optimizer


def run(lr, lam, batch_size, num_epoch, model):
    startTime = time.time()
    # Build computation graph
    if model == 'LR':
        X, y_target, y_predicted, mse, train = Training_lr(lr, lam)
    elif model == 'multi':
        X, y_target, y_predicted, mse, train = Training_multi(lr, lam)
    elif model == 'face':
        print("training face")
        X, y_target, y_predicted, mse, train = Training_multi_face(lr, lam)
    else:
        X, y_target, y_predicted, mse, train = Training(lr, lam, model)

    # Loading my data
    if model == 'multi':
        trainData, trainTarget, validData, validTarget, testData, testTarget = loaddata_multi()
        print("load multi class data")
    elif model == 'face':
        trainData, trainTarget, validData, validTarget, testData, testTarget = loaddata_face()
        print("load face data")
    else:
        trainData, trainTarget, validData, validTarget, testData, testTarget = loaddata()

    # Initialize session
    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init)

    loss_list = []
    loss_valid_list = []
    accuracy_list = []
    accuracy_valid_list = []
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
            sess.run(train, feed_dict={X: x, y_target: t})
            err, y_prediction = sess.run([mse, y_predicted], feed_dict={X: trainData, y_target: trainTarget})
            err_valid, y_prediction_valid = sess.run([mse, y_predicted], feed_dict={X: validData, y_target: validTarget})
            if model =='multi':
                training_accuracy = accuracy(trainTarget, y_prediction)
                training_accuracy_valid = accuracy(validTarget, y_prediction_valid)
            else:
                training_accuracy = accuracy_score(trainTarget, sign(y_prediction))
                training_accuracy_valid = accuracy_score(validTarget, sign(y_prediction_valid))
            # loss_list.append(err)

            if num_update < num_epoch:
                num_update += 1
            else:
                break
        duration = time.time() - startTime
        print('Epoch: {:4}, Loss: {:5f}, Accuracy:{:5f},Duration: {:2f}'.format(int(num_update/ num_steps), err, training_accuracy, duration))
        loss_list.append(err)
        loss_valid_list.append(err_valid)
        accuracy_list.append(training_accuracy)
        accuracy_valid_list.append(training_accuracy_valid)


    duration = time.time() - startTime

    # Test on the validation set
    valid_err, valid_predict = sess.run([mse, y_predicted], feed_dict={X: validData, y_target: validTarget})
    valid_predict = sign(valid_predict)
    # Testing model
    test_err, test_predict = sess.run([mse, y_predicted], feed_dict={X: testData, y_target: testTarget})
    test_predict = sign(test_predict)
    # accuracy
    valid_accuracy = accuracy_score(validTarget, valid_predict)
    test_accuracy = accuracy_score(testTarget, test_predict)

    print("valid MSE: %.4f" % valid_err)
    print("Valid Accuracy: %.7f" % valid_accuracy)
    print("test MSE: %.4f" % test_err)
    print("Test Accuracy: %.7f" % test_accuracy)

    print("Duration: %.4f" % duration)

    return loss_list,loss_valid_list,accuracy_list,accuracy_valid_list


def comparison():
    z = 0  # Dummy target
    x = np.linspace(0, 1, 100)
    y_mse = (x - z) ** 2
    y_cross = z * -np.log(x) + (1 - z) * -np.log(1 - x)
    plt.plot(x, y_mse, 'b', label='Mean Squared Error')
    plt.plot(x, y_cross, 'r', label='Cross Entropy Error')
    plt.legend()
    plt.ylabel("Mean Squared and Cross Entropy Errors")
    plt.xlabel("Deviation from Correct Label")
    plt.show()


def print_graph(question_number):
    if question_number == '2.1.1':
        loss,loss_v1, acc, acc_v1 = run(lr=0.01, lam=0.01, batch_size=500, num_epoch=5000, model='SGD')
        plt.figure(figsize=(15,10))
        plt.subplot(221)
        plt.title('Train Loss of Logistic Regression')
        plt.plot(loss, label="Learning Rate: 0.5", color='r')
        plt.ylabel('loss')
        plt.xlabel('num of epoch')
        plt.subplot(222)
        plt.title('validation Loss of Logistic Regression')
        plt.plot(loss_v1, label="Learning Rate: 0.5", color='r')
        plt.ylabel('loss')
        plt.xlabel('num of epoch')
        plt.subplot(223)
        plt.title('train accuracy of Logistic Regression')
        plt.plot(acc, label="Learning Rate: 0.5", color='b')
        plt.ylabel('accuracy')
        plt.xlabel('num of epoch')
        plt.subplot(224)
        plt.title('validation accuracy of Logistic Regression')
        plt.plot(acc_v1, label="Learning Rate: 0.5", color='b')
        plt.ylabel('accuracy')
        plt.xlabel('num of epoch')

        plt.show()
    if question_number == '2.1.2':
        loss_SGD,_,_,_ = run(lr=0.001, lam=0.01, batch_size=500, num_epoch=5000, model='SGD')
        loss_AO,_,_,_ = run(lr=0.001, lam=0.01, batch_size=500, num_epoch=5000, model='AO')
        plt.title('Loss of Logistic Regression for Different Optimizer')
        SGD, = plt.plot(loss_SGD, label="Optimizer: SGD", color='r')
        AO, = plt.plot(loss_AO, label="Optimizer: AO", color='b')
        plt.legend(handles=[SGD, AO], loc='upper right')
        plt.ylabel('loss')
        plt.xlabel('num of epoch')
        plt.show()
    if question_number == '2.1.3':
        loss_AO,_,acc_AO,_ = run(lr=0.001, lam=0, batch_size=500, num_epoch=5000, model='AO')
        loss_LR,_,acc_LR,_ = run(lr=0.001, lam=0, batch_size=500, num_epoch=5000, model='LR')
        plt.title('Loss of Logistic Regression and Linear Regression')
        AO, = plt.plot(loss_AO, label="Optimizer: AO", color='r')
        LR, = plt.plot(loss_LR, label="Optimizer: LR", color='b')
        plt.legend(handles=[AO, LR], loc='upper right')
        plt.ylabel('loss')
        plt.xlabel('num of epoch')
        plt.show()
        plt.title('Accuracy of Logistic Regression and Linear Regression')
        AO, = plt.plot(acc_AO, label="Optimizer: AO", color='r')
        LR, = plt.plot(acc_LR, label="Optimizer: LR", color='b')
        plt.legend(handles=[AO, LR], loc='upper right')
        plt.ylabel('Accuracy')
        plt.xlabel('num of epoch')
        plt.show()
        comparison()
    if question_number == '2.2.1':
        loss, loss_v1, acc, acc_v1 = run(lr=0.001, lam=0.01, batch_size=500, num_epoch=5000, model='multi')
        plt.figure(figsize=(15, 10))
        plt.subplot(221)
        plt.title('Train Loss of Multi-Class')
        plt.plot(loss, label="Learning Rate: 0.5", color='r')
        plt.ylabel('loss')
        plt.xlabel('num of epoch')
        plt.subplot(222)
        plt.title('validation Loss of Multi-Class')
        plt.plot(loss_v1, label="Learning Rate: 0.5", color='r')
        plt.ylabel('loss')
        plt.xlabel('num of epoch')
        plt.subplot(223)
        plt.title('train accuracy of Multi-Class')
        plt.plot(acc, label="Learning Rate: 0.5", color='b')
        plt.ylabel('accuracy')
        plt.xlabel('num of epoch')
        plt.subplot(224)
        plt.title('validation accuracy of Multi-Class')
        plt.plot(acc_v1, label="Learning Rate: 0.5", color='b')
        plt.ylabel('accuracy')
        plt.xlabel('num of epoch')
        plt.show()
    if question_number == '2.2.2':
        loss, loss_v1, acc, acc_v1 = run(lr=0.001, lam=0.01, batch_size=500, num_epoch=5000, model='face')
        plt.figure(figsize=(15, 10))
        plt.subplot(221)
        plt.title('Train Loss of Multi-Class')
        plt.plot(loss, label="Learning Rate: 0.5", color='r')
        plt.ylabel('loss')
        plt.xlabel('num of epoch')
        plt.subplot(222)
        plt.title('validation Loss of Multi-Class')
        plt.plot(loss_v1, label="Learning Rate: 0.5", color='r')
        plt.ylabel('loss')
        plt.xlabel('num of epoch')
        plt.subplot(223)
        plt.title('train accuracy of Multi-Class')
        plt.plot(acc, label="Learning Rate: 0.5", color='b')
        plt.ylabel('accuracy')
        plt.xlabel('num of epoch')
        plt.subplot(224)
        plt.title('validation accuracy of Multi-Class')
        plt.plot(acc_v1, label="Learning Rate: 0.5", color='b')
        plt.ylabel('accuracy')
        plt.xlabel('num of epoch')
        plt.show()


if __name__ == '__main__':

    # print_graph('2.1.1')
    # print_graph('2.1.2')
    # print_graph('2.1.3')
    print_graph('2.2.1')
    # print_graph('2.2.2')

    print('finish')