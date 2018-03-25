import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import Q1
from PIL import Image


def data_segmentation(task):
    """

    :param task: choose a mode 0 or 1
    :return: training data, validation data, test data
    """
    # task = 0 >> select the name ID targets for face recognition task
    # task = 1 >> select the gender ID targets for gender recognition task
    data = np.load('data.npy') / 255
    data = np.reshape(data, [-1, 32 * 32])
    target = np.load('target.npy')
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    trBatch = int(0.8 * len(rnd_idx))
    validBatch = int(0.1 * len(rnd_idx))
    trainData, validData, testData = data[rnd_idx[1:trBatch], :], \
                                     data[rnd_idx[trBatch + 1:trBatch + validBatch], :], \
                                     data[rnd_idx[trBatch + validBatch + 1:-1], :]
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
                                           target[rnd_idx[trBatch + 1:trBatch + validBatch], task], \
                                           target[rnd_idx[trBatch + validBatch + 1:-1], task]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def find_knn(test_x, train_x, train_y, k):
    """

    :param test_x: a sample to test
    :param train_x: the data for training
    :param train_y: the training target
    :param k: the number of nearest neighbours
    :return: the prediction target
    """

    # compute the distances between train and test data points
    distances = Q1.distanceFunc(test_x, train_x)
    neg_distance = - distances
    # take top k element
    _, indices = tf.nn.top_k(neg_distance, k=k)

    # build a N2 dim vector, with targets for the test data points
    shape = test_x.shape[0]
    prediction_y = tf.zeros([shape], tf.int32)

    # find the nearest neighbor of each point
    for i in range(shape):
        k_neighbors = tf.gather(train_y, indices[i, :])

        # find the most possible neighbor
        values, _, counts = tf.unique_with_counts(tf.reshape(k_neighbors, shape=[-1]))
        _, max_count_idx = tf.nn.top_k(counts, k=1)
        prediction = tf.gather(values, max_count_idx)

        # add the dense to the prediction set
        sparse_test_target = tf.SparseTensor([[i]], prediction, [shape])
        prediction_y = tf.add(prediction_y, tf.sparse_tensor_to_dense(sparse_test_target))
    return prediction_y


def accuracy(prediction_target, target):
    """

    :param prediction_target: perdiction target
    :param target: the true target
    :return: the accuracy rate
    """
    # find the points that are same as target answer
    true_points = tf.count_nonzero(tf.equal(prediction_target, target))
    # calculate the accuracy rate
    return true_points / target.shape[0]


def get_best_k(test_mode):
    """

    :param test_mode: choose a mode 0 or 1 to decide classification type

    """
    train_X, valid_X, test_X, train_Y, valid_Y, test_Y = data_segmentation(test_mode)

    train_data = tf.Variable(train_X, dtype=tf.float32)
    train_target = tf.Variable(train_Y, dtype=tf.int32)
    valid_data = tf.Variable(valid_X, dtype=tf.float32)
    valid_target = tf.Variable(valid_Y, dtype=tf.int32)
    test_data = tf.Variable(test_X, dtype=tf.float32)
    test_target = tf.Variable(test_Y, dtype=tf.int32)

    # interact
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Find the nearest k neighbours:
    ks = [1, 5, 10, 25, 50, 100, 200]
    # initialize the optimal solution
    max_valid = float("-inf")
    k_best = float("inf")
    # test each value of k
    for kc in ks:

        prediction_valid = find_knn(valid_data, train_data, train_target, kc)
        # accuracy of the
        acc = accuracy(prediction_valid, valid_target)
        a = sess.run(acc)
        print("The accuracy is {} with k = {}\n".format(a, kc))
        if a > max_valid or max_valid is None:
            max_valid = a
            k_best = kc
    # test with the k selected from validation
    prediction_test = find_knn(test_data, train_data, train_target, k_best)
    acc = accuracy(prediction_test, test_target)
    a = sess.run(acc)
    print("The accuracy is {} with k = {}\n".format(a, k_best))


    # misclassifications for K=10
    test_prediction = find_knn(test_data, train_data, train_target, 10)
    # find a misclassification
    mis_idx = tf.where(tf.not_equal(test_prediction, test_target))[0]
    # get the 10 nearest neighbor training data of this failed test case
    distances = Q1.distanceFunc(tf.gather(test_data, mis_idx), train_data)
    nearest_k_train_values, nearest_k_indices = tf.nn.top_k(-1 * distances, k=10)

    img = Image.fromarray(255 * sess.run(tf.reshape(tf.gather(test_data, mis_idx), [32, 32])))
    plt.imshow(img, cmap='gray')
    plt.show()
    for j in range(10):
        plt.subplot(2, 5, j + 1)
        img = Image.fromarray(255 * sess.run(tf.reshape(tf.gather(train_data, nearest_k_indices[:, j]), [32, 32])))
        plt.imshow(img, cmap='gray')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("face recognition:\n")
    get_best_k(0)
    print("=========================================================================")
    print("gender recognition:\n")
    get_best_k(1)
