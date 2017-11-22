import pickle 


def unpickle(f):
    with open(f, 'rb') as fo:
        dic = pickle.load(fo)
    print dic
    return dic


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
