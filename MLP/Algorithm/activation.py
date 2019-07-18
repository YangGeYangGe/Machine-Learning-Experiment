
import numpy as np
class Activation(object):
    def __tanh(self, x):
        return np.tanh(x)
    def __tanh_deriv(self, a):
        return 1.0 - a**2
    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    def __logistic_deriv(self, a):
        return  a * (1 - a )
#     relu
    def __relu(self, a):
        return np.maximum(0,a)
    def __relu_deriv(self, a):
        a[a>0]=1
        a[a<=0]=0
        return a
#     softmax
#     def __softmax(self, a):
#         e = np.exp(a)
#         return e / np.sum(e)
    
#     to avoid a very large x
    def __softmax(self, x):
# """Compute the softmax of vector x in a numerically stable way."""
        shiftx = x - np.max(x,axis = 1, keepdims=True)
        exps = np.exp(shiftx)
        return exps / np.sum(exps,axis = 1, keepdims=True)
    
    def __init__(self,activation='tanh'):
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv 
        elif activation == 'softmax':
            # derivative of softmax is combined with cross entropy in mlp class
            self.f = self.__softmax
            