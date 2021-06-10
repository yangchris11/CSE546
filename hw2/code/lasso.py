import numpy as np


class Lasso:
    def __init__(self, X, y, reg_lambda):
        self.reg_lambda = reg_lambda
        self.X = X
        self.y = y
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.delta = 1e-3
        self.global_w_history = {}
        self.local_w_history = []
        self.cur_iter = 0

    def reset(self, reg_lambda):
        self.reg_lambda = reg_lambda
        self.w = np.zeros(self.X.shape[1])
        self.b = 0
        self.local_w_history = []
        self.cur_iter = 0

    def fit(self):
        a = 2 * np.sum(self.X**2, axis=0)

        while self.cur_iter == 0 or np.linalg.norm(self.w - self.local_w_history[-1]) > self.delta:
            self.cur_iter += 1
            self.local_w_history.append(np.copy(self.w))
            # print('Iteration {}'.format(self.cur_iter))
            self.b = np.mean(self.y - self.X.dot(self.w))
            for k in range(self.X.shape[1]):
                a_k = a[k]
                c_k = 2 * np.sum(self.X[:,k] * (self.y - (self.b + self.X.dot(self.w) - self.X[:,k].dot(self.w[k]))), axis=0)
                if c_k < -self.reg_lambda:
                    self.w[k] = (c_k + self.reg_lambda) / a_k
                elif c_k > self.reg_lambda:
                    self.w[k] = (c_k - self.reg_lambda) / a_k
                else:
                    self.w[k] = 0

        self.global_w_history[self.reg_lambda] = self.w

    def loss(self):
        return (np.linalg.norm(self.X.dot(self.w) + self.b - self.y))**2 + \
                    self.reg_lambda * np.linalg.norm(self.w)

    def predict(self, X_pred):
        return X_pred.dot(self.w) + self.b

