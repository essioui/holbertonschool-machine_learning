#!/usr/bin/env python3
"""
    Build a deep neural network
"""
import numpy as np
import pickle


class DeepNeuralNetwork:
    """ Defines a deep neural network
        performing binary classification """

    def __init__(self, nx, layers):
        """ nx is number of input features
            must be an integer and >= 1
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)

        def checklist(li, i, L):
            """ checks if every element > 1 """
            if i == L:
                return
            if li[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            checklist(li, i + 1, L)

        checklist(layers, 0, self.__L)
        self.__cache = dict()
        self.__weights = dict()
        for i in range(1, self.__L + 1):
            if i == 1:
                l2 = nx
            else:
                l2 = layers[i - 2]
            l1 = layers[i - 1]
            w = 'W' + str(i)
            b = 'b' + str(i)
            self.__weights.update({w: np.random.randn(l1, l2)
                                  * np.sqrt(2 / l2)})
            self.__weights.update({b: np.zeros((l1, 1))})

    @property
    def L(self):
        """ L getter """
        return self.__L

    @property
    def cache(self):
        """ cache getter """
        return self.__cache

    @property
    def weights(self):
        """ weights getter """
        return self.__weights

    def forward_prop(self, X):
        """ Forward propagation of the network
            X contains the input data (nx, m)
            returns final output and activations cache
        """
        self.__cache['A0'] = X

        def sig_act(aw):
            """ sigmoid activation function """
            return 1 / (1 + np.exp(-aw))

        for i in range(self.L):
            w = 'W' + str(i + 1)
            a = 'A' + str(i)
            b = 'b' + str(i + 1)
            aw_ = np.matmul(self.__weights[w],
                            self.__cache[a]) + self.__weights[b]
            A = 'A' + str(i + 1)
            self.__cache[A] = sig_act(aw_)
        return self.__cache[A], self.__cache

    def cost(self, Y, A):
        """ calculates the cost of the model using logistic regression
            Y contains the correct labels
            A contains the activated output for each example
        """
        m = Y.shape[1]
        loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = -(1 / m) * loss
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural networks predictions
            X contains the input data (nx, m)
            Y contains the correct labels
        """
        a, _ = self.forward_prop(X)
        cost = self.cost(Y, a)
        x = np.where(a >= 0.5, 1, 0)
        return x, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the NN
            Y contains correct labels
        """
        m = Y.shape[1]
        wold = self.weights.copy()
        for i in range(self.L, 0, -1):
            A_i = cache['A' + str(i)]
            A_iless1 = cache['A' + str(i - 1)]
            if i == self.L:
                dz = np.subtract(A_i, Y)
            else:
                dz = np.matmul(wold['W' + str(i + 1)].T, dz2) *\
                     np.multiply(A_i, (1 - A_i))
            dz2 = dz
            dw = np.matmul(dz, A_iless1.T) / m
            wupdate = np.subtract(wold['W' + str(i)], np.multiply(alpha, dw))
            bupdate = np.subtract(self.weights['b' + str(i)],
                                  np.multiply(alpha, np.sum(dz,
                                              axis=1, keepdims=True) / m))
            self.weights['W' + str(i)] = wupdate
            self.weights['b' + str(i)] = bupdate

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Trins the deep neural network """
        import matplotlib.pyplot as plt
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            A2, cache = self.forward_prop(X)
            if i == 0:
                c = self.cost(Y, A2)
                cost = [c]
                steps = [0]
                if verbose:
                    print("Cost after {} iterations: {}".format(i, c))
            elif i % step == 0:
                c = self.cost(Y, A2)
                cost.append(c)
                steps.append(i)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, c))
            self.gradient_descent(Y, cache, alpha)
        i += 1
        if verbose:
            print("Cost after {} iterations: {}".format(i, c))
        if graph:
            plt.plot(steps, cost)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """ Saves the instance object to a file in picke format
            add .pkl extension to filename
        """
        if filename[-4:] != '.pkl':
            filename += ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()

    @staticmethod
    def load(filename):
        """ Loads picked deepnn object
            Returns loaded obj or None
        """
        try:
            with open(filename, 'rb') as f:
                pick = pickle.load(f)
                f.close()
            return pick
        except Exception:
            return None
