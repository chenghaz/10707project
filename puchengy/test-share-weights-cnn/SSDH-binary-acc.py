import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='PyTorch SSDH-binary Prediction')
parser.add_argument('--binary_path', default='./checkpoint/binary_code_128', type=str)
parser.add_argument('--output_path', default='./checkpoint/binary_code_predict_128', type=str)
parser.add_argument('--top', default=50, type=int)
args = parser.parse_args()

params = pickle.load(open(args.binary_path, 'r'))
test = params['binary_code_test']
train = params['binary_code_train']

top = args.top
aps = []
for i in range(len(test)):
    print(i)
    test_labels = np.tile(test[i, -1], (len(train), ))
    test_features = np.tile(test[i, :-1], (len(train), 1))
    train_features = train[:, :-1]
    train_labels = train[:, -1]
    distance = np.reshape(np.count_nonzero(test_features != train_features, 1), (len(train), 1))
    results = np.reshape(test_labels == train_labels, (len(train), 1))
    distance_results = np.concatenate((distance, results), 1)
    distance_results = distance_results[distance_results[:, 0].argsort()]
    top_results = distance_results[: top, -1]
    # AP
    ap = []
    count = 0
    correct = 0
    for i in range(top):
        count += 1
        if top_results[i]:
            correct += 1
            ap.append(correct * 1.0 / count)
    if len(ap) != 0:
        aps.append(sum(ap) * 1.0 / len(ap))
    else:
        aps.append(0.0)
print(args.binary_path)
print("MAP is:" + str(sum(aps) * 1.0 / len(aps)))