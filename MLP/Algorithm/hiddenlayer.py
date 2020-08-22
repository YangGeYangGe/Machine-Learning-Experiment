
from activation import Activation
import numpy as np

class Batchnorm():
    def __init__(self,X_dim):
        self.gamma = np.ones((1, X_dim))
        self.beta = 0
        # record history
        self.mu_list = []
        self.var_list = []
        
        self.dgamma = 0
        self.dbeta = 0
    def forward(self,X):
        self.X = X
        self.mu = np.mean(X,axis = 0)
        self.var = np.var(X, axis=0)
        # update history
        self.mu_list.append(self.mu)
        # self.var_list = (self.var)
        self.var_list.append(self.var)
        self.X_norm = (X - self.mu)/np.sqrt(self.var + 1e-8)
        out =  self.X_norm*self.gamma + self.beta
        
        return out

    def backward(self,dout):
#         https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
        X = self. X
        X_norm = self.X_norm
        mu = self.mu
        var = self.var
        gamma = self.gamma
        beta = self.beta
        
        N, D = X.shape

        X_mu = X - mu
        std_inv = 1. / np.sqrt(var + 1e-8)

        dX_norm = dout * gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
        dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

        dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
        dgamma = np.sum(dout * X_norm, axis=0)
        dbeta = np.sum(dout, axis=0)

        self.dgamma = dgamma
        self.dbeta = dbeta
        return dX
    # calculate average mu and var for predicting
    def prepare_predict(self,batch_size):
        self.mu = np.mean(self.mu_list,axis = 0)
        self.var = batch_size/(batch_size-1) * np.mean(self.var_list, axis = 0)
        

    def forward_predict(self, X):
        self.X = X
        self.X_norm = (X - self.mu)/np.sqrt(self.var + 1e-8)
        out =  self.X_norm*self.gamma + self.beta
        return out

class HiddenLayer(object):    
    def __init__(self,n_in, n_out,
                 activation_last_layer='relu',activation='relu', W=None, b=None):

        self.input=None
        self.activation=Activation(activation).f

        self.activation_deriv=None
        if activation_last_layer:
            self.activation_deriv=Activation(activation_last_layer).f_deriv

        self.W = np.random.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
        )
        if activation == 'logistic':
            self.W *= 4

        self.b = np.zeros(n_out,)
        
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)
        
#         for dropout, self.dropout is keep prob, u1 would be np.random.binomial 
        self.dropout = 1
        self.u1 = None
        
#         for momentum
        self.v_w = np.zeros(self.W.shape)
        self.v_b = np.zeros(self.b.shape)

#         for batchnorm
        self.bn = Batchnorm(n_out)
    
    # forward for training
    def forward(self, input, dropout= 1, output_layer=False,batch_norm = False):
        input = input.reshape(-1, self.W.shape[0])   

        self.batch_norm = batch_norm
        lin_output = np.dot(input, self.W) + self.b

        #for batchnorm
        if batch_norm:
            lin_output = self.bn.forward(lin_output)
            
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        #dropout
        if (dropout != 1) and (output_layer == False):
            # 除以dropout probability它会大致矫正或者补足你丢失的dropout%，以确保output的期望值仍然维持在同一水准
            u1 = np.random.binomial(1, dropout, size= self.output.shape[1])/dropout
            self.output *= u1
            self.u1 = u1
            self.dropout = dropout

        self.input=input
        return self.output
     
    def backward(self, delta, output_layer=False):   
        # batchnorm backward   
        if self.batch_norm:
            delta= self.bn.backward(delta)                
   
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))

        # dropout backward
        if self.dropout != 1:
            # self.grad_W *= self.u1/self.dropout
            self.grad_W *= self.u1
        self.grad_b = np.sum(delta,axis = 0)
        
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
        return delta
    
    # forward for predcting
    def forward_predict(self, input):
        input = input.reshape(-1, self.W.shape[0])
        lin_output = np.dot(input, self.W) + self.b

        # batchnorm forward for predicting
        if self.batch_norm:
            lin_output = self.bn.forward_predict(lin_output)
            
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input=input
        return self.output


    