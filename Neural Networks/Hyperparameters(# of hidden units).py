import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


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


def Train(l_rate, Lambda):
    W = {
        'layer_1': tf.Variable(tf.random_normal([n_input, n_hidden_1], mean=0.0, stddev=3.0 / (n_input + n_hidden_1))),
        'output': tf.Variable(tf.random_normal([n_hidden_1, n_classes], mean=0.0, stddev=3.0 / (n_hidden_1 + n_classes)))
    }
    b = {
        'Layer_1': tf.Variable(tf.random_normal([n_hidden_1])),
        'output': tf.Variable(tf.random_normal([n_classes]))
    }
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32, [None, 10], name='target_y')

    # Graph definition
    y_predicted_1 = tf.nn.relu(tf.add(tf.matmul(X, W['layer_1']), b['Layer_1']))
    y_predicted = tf.matmul(y_predicted_1,W['output'])+b['output']
    # Error definition
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y_predicted)
                          , name='mean_squared_error') + tf.reduce_sum(tf.square(W['layer_1'])) * Lambda / 2

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(loss)
    return X, y_target, y_predicted, loss, optimizer


def run(lr, lam, batch_size, num_epoch):

    startTime = time.time()
    # Build computation graph
    X, y_target, y_predicted, mse, train = Train(lr, lam)

    # Loading my data
    trainData, trainTarget, validData, validTarget, testData, testTarget = loaddata_multi()
    print("load multi class data")

    # Initialize session
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    sess.run(init)

    loss_list = []
    loss_valid_list = []
    loss_test_list = []
    accuracy_list = []
    accuracy_valid_list = []
    accuracy_test_list = []
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
            err_test, y_prediction_test = sess.run([mse, y_predicted], feed_dict={X: testData, y_target: testTarget})
            training_accuracy = accuracy(trainTarget, y_prediction)
            training_accuracy_valid = accuracy(validTarget, y_prediction_valid)
            training_accuracy_test = accuracy(testTarget,y_prediction_test)
            if num_update < num_epoch:
                num_update += 1
            else:
                break
        duration = time.time() - startTime
        print('Epoch: {:4},Train Loss: {:5f},Train Accuracy:{:5f},Validation Loss:{:5f},Validation Accuracy:{:5f} '
              'Duration: {:2f}'.format(int(num_update/ num_steps), err, training_accuracy, err_valid,
                                       training_accuracy_valid, duration))
        loss_list.append(err)
        loss_valid_list.append(err_valid)
        loss_test_list.append(err_test)
        accuracy_list.append(training_accuracy)
        accuracy_valid_list.append(training_accuracy_valid)
        accuracy_test_list.append(training_accuracy_test)
        '''
        i = int(num_update / num_steps)
        if i==5 or i==10 or i==15 or i==20:
            save_path = saver.save(sess, "/home/horis/Documents/Machine Learning examples/Assignment 3/Feedforward_models/model" + str(i) + ".ckpt")
            print("Model saved in file: ", save_path)
        '''
    duration = time.time() - startTime

    # Test on the validation set
    valid_err, valid_predict = sess.run([mse, y_predicted], feed_dict={X: validData, y_target: validTarget})
    # Testing model
    test_err, test_predict = sess.run([mse, y_predicted], feed_dict={X: testData, y_target: testTarget})
    # accuracy
    valid_accuracy = accuracy(validTarget, valid_predict)
    test_accuracy = accuracy(testTarget, test_predict)

    print("valid MSE: %.4f" % valid_err)
    print("Valid Accuracy: %.7f" % valid_accuracy)
    print("test MSE: %.4f" % test_err)
    print("Test Accuracy: %.7f" % test_accuracy)

    print("Duration: %.4f" % duration)

    return loss_list,loss_valid_list,loss_test_list, accuracy_list,accuracy_valid_list, accuracy_test_list


def print_graph(lr, lam, batch_size, num_epoch):

    loss_training, loss_validation, loss_test, acc, acc_validation, acc_test = run(lr, lam, batch_size, num_epoch)
    plt.figure()
    plt.title("Loss of neural networks")
    plt.plot(loss_training, label="Training Set")
    plt.plot(loss_validation, label="Validation Set")
    plt.plot(loss_test, label="Test Set")
    plt.legend(loc='upper right')

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    plt.figure()
    plt.title("Accuracy of neural networks")
    plt.plot(acc, label="Training Set")
    plt.plot(acc_validation, label="Validation Set")
    plt.plot(acc_test, label="Test Set")
    plt.legend(loc='lower right')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()


if __name__ == '__main__':

    # Parameters
    learning_rate = 0.001
    training_epochs = 1000
    batch_size = 500
    lam = 3e-4

    # Network Parameters
    n_hidden_array = [100, 500, 1000]  # 1st layer number of features
    n_input = 784  # MNIST data input (img shape: 28*28)
    n_classes = 10  # MNIST total classes (0-9 digits
    for i in range(3):
        n_hidden_1 = n_hidden_array[i]
        print_graph(learning_rate, lam, batch_size, training_epochs)
    print('finish')