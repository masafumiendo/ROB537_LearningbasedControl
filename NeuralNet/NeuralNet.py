import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

class NeuralNet:

    # Constructor
    def __init__(self, node_input, node_hidden, node_output):

        self.weight_hidden = np.random.random_sample((node_hidden, node_input+1))
        self.weight_output = np.random.random_sample((node_output, node_hidden+1))
        self.momentum_hidden = np.zeros((node_hidden, node_input+1))
        self.momentum_output = np.zeros((node_output, node_hidden+1))

    # Public method
    def train(self, X_train, T_train, X_valid, T_valid, batch_size, learning_rate, mu, epoch):

        error_train = np.zeros(epoch)
        error_valid = np.zeros(epoch)

        # Loop for updating weight
        for i in range(epoch):
            p = np.random.permutation(len(X_train))
            X_train = X_train[p]
            T_train = T_train[p]
            q = np.random.permutation(len(X_valid))
            X_valid = X_valid[q]
            T_valid = T_valid[q]
            for j in range(batch_size):
                x = X_train[j, :]
                t = T_train[j, :]

                weight_output_optimized, weight_hidden_optimized = self.__backprop(x, t, learning_rate, mu)

            error_train[i] = self.__calc_error(X_train, T_train)
            error_valid[i] = self.__calc_error(X_valid, T_valid)

        return weight_output_optimized, weight_hidden_optimized, error_train, error_valid

    def predict(self, X, T, weight_output_optimized, weight_hidden_optimized):

        N = X.shape[0]
        C = np.zeros(N).astype('int')
        Y = np.zeros((N, X.shape[1]))
        _T = np.zeros(N).astype('int')

        for i in range(N):

            x = X[i, :]
            z = self.__sigmoid(weight_hidden_optimized.dot(np.r_[np.array([1]), x]))
            y = self.__sigmoid(weight_output_optimized.dot(np.r_[np.array([1]), z]))

            t = T[i, :]
            if t[0] == 0:
                _T[i] = 1
            elif t[0] == 1:
                _T[i] = 0

            Y[i] = y
            C[i] = y.argmax()

        TPN = 0

        for j in range(N):

            if C[j] == _T[j]:
                TPN += 1
            else:
                pass

        acc = TPN / N

        return acc

    def error_graph(self, figname, error_train, error_valid):

        plt.figure()
        plt.plot(np.arange(0, error_train.shape[0]), error_train, label='training error')
        plt.plot(np.arange(0, error_valid.shape[0]), error_valid, label='validation error')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('./fig/' + figname)

    # Private method
    def __sigmoid(self, arr):

        return np.vectorize(lambda x: 1.0 / (1.0 + math.exp(-x)))(arr)

    def __forward(self, x):

        # Calculate output (z: output from hidden layer, y: output from output layer)
        z = self.__sigmoid(self.weight_hidden.dot(np.r_[np.array([1]), x]))
        y = self.__sigmoid(self.weight_output.dot(np.r_[np.array([1]), z]))

        return z, y

    def __backprop(self, x, t, learning_rate, mu):

        # Calculate output
        z, y = self.__forward(x)

        # Update output weight with momentum
        delta_output = (y - t) * y * (1.0 - y) # y * (1.0 - y) is a derivative of sigmoid function
        _weight_output = self.weight_output
        self.weight_output -= learning_rate * delta_output.reshape((-1, 1)) * np.r_[np.array([1]), z] - mu * self.momentum_output
        self.momentum_output = self.weight_output - _weight_output

        # Update hidden weight with momentum
        delta_hidden = (self.weight_output[:, 1:].T.dot(delta_output)) * z * (1.0 - z) # z * (1.0 - z) is a derivative of sigmoid function
        _weight_hidden = self.weight_hidden
        self.weight_hidden -= learning_rate * delta_hidden.reshape((-1, 1)) * np.r_[np.array([1]), x] - mu * self.momentum_hidden
        self.momentum_hidden = self.weight_hidden - _weight_hidden

        weight_output_optimized = self.weight_output
        weight_hidden_optimized = self.weight_hidden

        return weight_output_optimized, weight_hidden_optimized

    def __calc_error(self, X, T):

        N = X.shape[0]
        error = 0.0

        for i in range(N):

            x = X[i, :]
            t = T[i, :]

            z, y = self.__forward(x)
            error += (y - t).dot((y - t).reshape((-1, 1))) / (2.0 * N)

        return error

def pipeline(N, train_x, train_y, test_x, test_y, node_input, node_hidden, node_output, batch_size, learning_rate, mu, epoch, split_size):

    split_piece = int(N / split_size)
    Acc = np.array([])
    Weight_output = np.array([])
    Weight_hidden = np.array([])
    Error_train = np.array([])
    Error_valid = np.array([])

    print("Start neural network training with ", node_hidden, "'s hidden node, ", batch_size, "'s batch size, ", learning_rate, "'s learning rate, ", mu, "'s mu, ", epoch, "'s epoch, and ", split_size, "'s split size")

    for i in range(split_size):

        _train_x = np.delete(train_x, np.s_[(i) * split_piece:(i + 1) * split_piece], axis=0)
        _train_t = np.delete(train_y, np.s_[(i) * split_piece:(i + 1) * split_piece], axis=0)
        _valid_x = train_x[(i) * split_piece:(i + 1) * split_piece]
        _valid_t = train_y[(i) * split_piece:(i + 1) * split_piece]

        neuralnet_cv = NeuralNet(node_input, node_hidden, node_output)

        weight_output_optimized, weight_hidden_optimized, error_train, error_valid = neuralnet_cv.train(_train_x,
                                                                                                        _train_t,
                                                                                                        _valid_x,
                                                                                                        _valid_t,
                                                                                                        batch_size,
                                                                                                        learning_rate,
                                                                                                        mu, epoch)
        acc = neuralnet_cv.predict(_valid_x, _valid_t, weight_output_optimized, weight_hidden_optimized)
        Acc = np.append(Acc, acc)
        Weight_output = np.append(Weight_output, weight_output_optimized)
        Weight_hidden = np.append(Weight_hidden, weight_hidden_optimized)
        Error_train = np.append(Error_train, error_train)
        Error_valid = np.append(Error_valid, error_valid)

    print("Cross validation result: ", Acc)

    best_acc = np.argmax(Acc)

    Weight_output.reshape([-1, 4])
    Weight_hidden.reshape([-1, 3])
    Error_train.reshape([-1, 1])
    Error_valid.reshape([-1, 1])

    Weight_output = np.reshape(Weight_output, ([split_size, -1]))
    Weight_hidden = np.reshape(Weight_hidden, ([split_size, -1]))
    Error_train = np.reshape(Error_train, ([split_size, -1]))
    Error_valid = np.reshape(Error_valid, ([split_size, -1]))

    weight_output_optimized = np.reshape(Weight_output[best_acc], [node_output, -1])
    weight_hidden_optimized = np.reshape(Weight_hidden[best_acc], [node_hidden, -1])

    neuralnet = NeuralNet(node_input, node_hidden, node_output)

    neuralnet.error_graph('error_graph_'+str(node_hidden)+'hl_'+str(batch_size)+'bs_'+
                          str(learning_rate)+'lr_'+str(mu)+'mu_'+str(epoch)+'epoch_'+str(split_size)+'sz.png',
                          Error_train[best_acc], Error_valid[best_acc])

    acc = neuralnet.predict(test_x, test_y, weight_output_optimized, weight_hidden_optimized)

    summary = np.array([[node_hidden, batch_size, learning_rate, mu, epoch, split_size, acc]])

    print("Accuracy: ", acc)
    print("Summary is: ", summary)

    return summary

def main():

    train = pd.read_csv('./data/train1.csv', header=None).values
    train_x, train_y = np.split(train, 2, axis=1)
    test = pd.read_csv('./data/test1.csv', header=None).values
    test_x, test_y = np.split(test, 2, axis=1)

    N = train.shape[0] # Number of data

    node_input = train_x.shape[1]
    Node_hidden = [2, 4, 100]
    node_output = 2
    Batch_size = [1, 10, 100]
    Learning_rate = [0.01, 0.1, 0.9]
    Mu = [0., 0.5, 0.9]
    Epoch = [500, 1000, 2000]
    Split_size = [4]
    Summary = np.array([[1., 1., 1., 1., 1., 1., 1.]])

    # Shuffle data for cross validation
    p = np.random.permutation(len(train_x))
    train_x, train_y = train_x[p], train_y[p]

    for node_hidden in Node_hidden:
        for batch_size in Batch_size:
            for learning_rate in Learning_rate:
                for mu in Mu:
                    for epoch in Epoch:
                        for split_size in Split_size:
                            summary = pipeline(N, train_x, train_y, test_x, test_y, node_input, node_hidden, node_output,
                                      batch_size, learning_rate, mu, epoch, split_size)
                            Summary = np.append(Summary, summary, axis=0)

    columns = ['hidden nodes', 'batch size', 'learning rate', 'mu', 'epoch', 'split size', 'accuracy']
    Summary = pd.DataFrame(data=Summary, columns=columns, dtype=float)
    Summary.to_csv('summary.csv')

if __name__ == '__main__':
    main()