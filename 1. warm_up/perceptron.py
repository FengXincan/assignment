'''
report:
reference:
    https://medium.com/@uttam94/perception-implementation-from-scratch-using-python-90768afa505b
our difference:
    data is not the same
'''

# read train/test data into sentences
from audioop import bias
import numpy as np 


def read_data(path):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        sentences = []
        for item in range(len(lines)):
            sentences.append(lines[item].rstrip('\n').split('\t'))
    return data

y_train, x_train = read_data('titles-en-train.labeled')
x_test = read_data('titles-en-test.word')

class perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None
    def fit(self, x, y):
        n_samples, n_features = x.shape

        # init parameters weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i>0 else 0 for i in y])
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(x):
                # linear = w * x + b
                linear_output = np.dot(x_i, self.weights) + self.bias
                # y_pred = activation_func(linear)
                y_predicted = self.activation_func(linear_output)

                # perception update rule
                # update = learning_rate * (y_actual - y_pred)
                update = self.lr * (y_[idx] - y_predicted)
                # update the weight matrix
                self.weights += update * x_i 
                self.bias += update

    def predict(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    p.fit(x_train, y_train)
    predictions = p.predict(x_test)
    print('perceptron prediction accuracy: ', accuracy(y_test, predictions))

