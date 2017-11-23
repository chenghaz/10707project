import pickle
import numpy as np

params = pickle.load(open('/Users/pucheng/Desktop/final-project/10707project/puchengy/test-share-weights-cnn/pytorch-cifar/checkpoint/binary_code', 'r'))
code = params['code']
top = 50

correct = 0
for i in range(len(code)):
    print(i)
    labels = np.tile(code[i, -1], (top, ))
    features = np.tile(code[i, :-1], (len(code) - 1, 1))
    rest_features = code[np.delete(range(len(code)), i, 0), :-1]
    rest_labels = np.reshape(code[np.delete(range(len(code)), i, 0), -1], (len(code) - 1, 1))
    distance = np.reshape(np.count_nonzero(features != rest_features, 1), (len(code) - 1, 1))
    distance_labels = np.concatenate((distance, rest_labels), 1)
    distance_labels = distance_labels[distance_labels[:, 0].argsort()]
    top_labels = distance_labels[: top, -1]
    r = (labels == top_labels)
    correct += np.sum(r)
print(correct * 1.0 / (top * len(code)))