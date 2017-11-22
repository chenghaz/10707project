import pickle 
import numpy as np
from numpy.random import permutation
from tempfile import TemporaryFile


def unpickle(f):
    with open(f, 'rb') as fo:
        dic = pickle.load(fo)
    print dic
    return dic


def positive_pair(x, y, count):
    pair1 = np.empty([count, 3072])
    pair2 = np.empty([count, 3072])

    perm = permutation(count)
    x = x[perm]
    y = y[perm]
    tik = 0
    for i in range(x.shape[0]):
        for j in range(i + 1, y.shape[0]):
            if y[i] == y[j]:
                pair1[tik] = x[i]
                pair2[tik] = x[j]
                tik = tik + 1
            if tik == count:
                return pair1, pair2


def negative_pair(x, y, count):
    pair1 = np.empty([count, 3072])
    pair2 = np.empty([count, 3072])

    perm = permutation(count)
    x = x[perm]
    y = y[perm]
    tik = 0
    for i in range(x.shape[0]):
        for j in range(i + 1, y.shape[0]):
            if y[i] != y[j]:
                pair1[tik] = x[i]
                pair2[tik] = x[j]
                tik = tik + 1
            if tik == count:
                return pair1, pair2


train_x = []
train_y = []
train_batch = unpickle('/Users/chenghaz/Desktop/17Fall/10707/project/model/pytorch-cifar/data/cifar-10-batches-py/data_batch_1')
train_x.extend(train_batch['data'])
train_y.extend(train_batch['labels'])
train_batch = unpickle('/Users/chenghaz/Desktop/17Fall/10707/project/model/pytorch-cifar/data/cifar-10-batches-py/data_batch_2')
train_x.extend(train_batch['data'])
train_y.extend(train_batch['labels'])
train_batch = unpickle('/Users/chenghaz/Desktop/17Fall/10707/project/model/pytorch-cifar/data/cifar-10-batches-py/data_batch_3')
train_x.extend(train_batch['data'])
train_y.extend(train_batch['labels'])
train_batch = unpickle('/Users/chenghaz/Desktop/17Fall/10707/project/model/pytorch-cifar/data/cifar-10-batches-py/data_batch_4')
train_x.extend(train_batch['data'])
train_y.extend(train_batch['labels'])
train_batch = unpickle('/Users/chenghaz/Desktop/17Fall/10707/project/model/pytorch-cifar/data/cifar-10-batches-py/data_batch_5')
train_x.extend(train_batch['data'])
train_y.extend(train_batch['labels'])

test_x = []
test_y = []
test_batch = unpickle('/Users/chenghaz/Desktop/17Fall/10707/project/model/pytorch-cifar/data/cifar-10-batches-py/test_batch')
test_x.extend(test_batch['data'])
test_y.extend(test_batch['labels'])

train_x = np.asarray(train_x, dtype=np.float32)
train_y = np.asarray(train_y, dtype=np.float32)
test_x = np.asarray(test_x, dtype=np.float32)
test_y = np.asarray(test_y, dtype=np.float32)

n_train = 50000
n_test = 10000

train_pos_pair1, train_pos_pair2 = positive_pair(train_x, train_y, n_train)
np.savez('train_pos', train_pos_pair1, train_pos_pair2)
test_pos_pair1, test_pos_pair2 = positive_pair(test_x, test_y, n_test)
train_neg_pair1, train_neg_pair2 = negative_pair(train_x, train_y, n_train)
test_neg_pair1, test_neg_pair2 = negative_pair(test_x, test_y, n_test)

np.savez('train_pos', train_pos_pair1, train_pos_pair2)
np.savez('train_neg', train_neg_pair1, train_neg_pair2)
np.savez('test_pos', test_pos_pair1, test_pos_pair2)
np.savez('test_neg', test_neg_pair1, test_neg_pair2)


