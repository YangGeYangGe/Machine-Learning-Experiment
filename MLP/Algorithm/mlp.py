from hiddenlayer import HiddenLayer 
import numpy as np
from activation import Activation

class MLP:
    
    def __init__(self, layers, activation=[None,'tanh','tanh']):
        ### initialize layers
        self.layers=[]
        self.params=[]
        
        self.activation=activation
        for i in range(len(layers)-1):
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1]))
    # forward for training
    def forward(self,input):
        for i in range(len(self.layers)):
            if i != len(self.layers)-1:
                output=self.layers[i].forward(input,self.dropout,batch_norm = self.batch_norm)
            else:
                # not using dropout and batchnorm in output layer 
                output=self.layers[i].forward(input,self.dropout,output_layer=True)
            input=output
        return output
    # forward for predicting
    def forward_predict(self,input):
        for layer in self.layers:
            output=layer.forward_predict(input)
            input=output
        return output
    
#     mean squared error
    def criterion_MSE(self,y,y_hat):
        activation_deriv=Activation(self.activation[-1]).f_deriv
        error = y-y_hat
        loss=error**2
        # calculate the delta of the output layer
        delta=-error*activation_deriv(y_hat)
        # return loss and delta
        loss = np.sum(loss)
        return loss,delta
    
#     cross entropy loss, delta combined with softmax derivative
    def criterion_cross_entropy(self, y, y_hat):
        
        loss =  0 - np.sum(y * np.log(y_hat+1e-10),axis=1,keepdims= True)
#         loss = -np.sum(y * np.log(y_hat+1e-10))
#         derivative: https://deepnotes.io/softmax-crossentropy
        delta = (y_hat-y)/y.shape[0]
        return loss, delta
    
    # backward
    def backward(self,delta):
        delta=self.layers[-1].backward(delta,output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta=layer.backward(delta)

    def update(self,lr):
        for layer in self.layers:
            # used weight decay https://blog.csdn.net/qq_19918373/article/details/64904708
            # also updating batchnorm gamma and beta

            if self.momentum_gamma == 0:
                layer.W -= (lr * layer.grad_W + lr*self.weight_decay*layer.W)
                layer.b -= lr * layer.grad_b
                layer.bn.gamma -= lr * layer.bn.dgamma
                layer.bn.beta -= lr * layer.bn.dbeta

            # using momentum, usually momentum gamma is 0.9
            else:
                layer.v_w = self.momentum_gamma * layer.v_w + lr * layer.grad_W
                layer.W -= (layer.v_w + lr*self.weight_decay*layer.W)
                layer.v_b = self.momentum_gamma * layer.v_b + lr * layer.grad_b
                layer.b -= layer.v_b
                
                layer.bn.gamma -= lr * layer.bn.dgamma
                layer.bn.beta -= lr * layer.bn.dbeta
    
    # mini batches, return a list of batches index
    def mini_batches_random(self, X, y, mini_batch_size):
        num_samples = X.shape[0]
        permutation = list(np.random.permutation(num_samples))
        idx = []
        num_complete = int(num_samples / mini_batch_size)
        for i in range(num_complete):
            idx.append(permutation[i * mini_batch_size: (i + 1) * mini_batch_size])
        if num_samples % mini_batch_size != 0:
            idx.append(permutation[num_complete * mini_batch_size:])
        return idx

    def fit(self,X,y,learning_rate=0.1, epochs=100,dropout = 1,batch_size = 0,weight_decay = 0, momentum_gamma=0,batch_norm = False,loss_function = "cross_entropy"):
        X=np.array(X)
        y=np.array(y)
        to_return = np.zeros(epochs)
        # for dropout (keep prob)
        self.dropout = dropout
        # for weight decay
        self.weight_decay = weight_decay
        # for momentum
        self.momentum_gamma = momentum_gamma
        #for batch norm
        self.batch_norm = batch_norm

        for k in range(epochs):
            print("epochs = ", k)
            
            # sgd
            if batch_size == 0:
                # X.shape[0]
                tmp_int = 10000
                loss=np.zeros(tmp_int)
                for it in range(tmp_int):
                    #random integers
                    i=np.random.randint(X.shape[0])                
                    # forward pass
                    y_hat = self.forward(X[i])
                    if loss_function == "MSE":
                        loss[it],delta=self.criterion_MSE(y[i],y_hat)
                        self.backward(delta)
                    elif loss_function == "cross_entropy":
                        loss[it],delta=self.criterion_cross_entropy(y[i],y_hat)
                        self.backward(delta)
                    # update
                    self.update(learning_rate)
                to_return[k] = np.mean(loss)
                print(to_return[k])

            # mini batche
            elif batch_size != 0:
                mini_batches = self.mini_batches_random(X, y, batch_size)
                loss=np.zeros(len(mini_batches))
                for i in range(len(mini_batches)):
                    y_hat = self.forward(X[mini_batches[i]])
                    if loss_function == "MSE":
                        loss[it],delta=self.criterion_MSE(y[mini_batches[i]],y_hat)
                        self.backward(delta)   
                    elif loss_function == "cross_entropy":
                        loss_it,delta=self.criterion_cross_entropy(y[mini_batches[i]],y_hat)
                        self.backward(delta)
                    # update
                    self.update(learning_rate)
                    loss[i] = np.mean(loss_it)    
                to_return[k] = np.mean(loss)
                print(to_return[k])

        # if using batchnorm, after training, calculate average mu and var for predicting.
        if batch_norm:
            for i in range(len(self.layers)):
                if i != len(self.layers)-1:
                    self.layers[i].bn.prepare_predict(batch_size)
        return to_return
        
    # predicting
    def predict(self, x):
        x = np.array(x)
        output = np.zeros((x.shape[0],self.layers[-1].b.shape[0]))
        for i in np.arange(x.shape[0]):
            output[i] = self.forward_predict(x[i,:])
        return output