#%%
#!/usr/bin/env python3
import os
import numpy as np
from numpy import random
import scipy
import matplotlib 
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot as plt

from scipy.special import softmax

mnist_data_directory = os.path.join(os.path.dirname('__file__'), "data")

# TODO add any additional imports and global variables
import time
from tqdm import tqdm 
#%%
def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = np.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        np.random.seed(8675309)
        perm = np.random.permutation(60000)
        Xs_tr = np.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = np.ascontiguousarray(Ys_tr[:,perm])
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = np.ascontiguousarray(Xs_te)
        Ys_te = np.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the gradient of the multinomial logistic regression objective, with regularization (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    # TODO students should use their implementation from programming assignment 2
    d,n = Xs.shape
    c,_ = Ys.shape
    size = len(ii)
    x = np.reshape(Xs[:,ii],(d,size))
    y = np.reshape(Ys[:,ii],(c,size))
    L_grad = np.dot(softmax(np.dot(W,x),axis=0)-y, x.T)                            
    L = L_grad/size + gamma*W
    return L

# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should use their implementation from programming assignment 1
    d, n = Xs.shape
    yTr = np.argmax(np.dot(W,Xs), axis=0)
    yLabel = np.argmax(Ys, axis=0)
    err = len(np.argwhere(yTr-yLabel))
    return err/n


# compute the cross-entropy loss of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss(Xs, Ys, gamma, W):
    # TODO students should implement this
    (d,n) = Xs.shape
    acc = 0.0
    acc = (-1.0/n)*np.sum(np.multiply(Ys, (np.log(softmax(np.dot(W,Xs), axis=0))))) + gamma/2*np.sum(W*W)
    return acc

# gradient descent (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
# monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" epochs
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    # TODO students should use their implementation from programming assignment 1
    d, n = Xs.shape
    count = 0
    weight = []
    ii = np.arange(n)
    for i in tqdm(range(num_epochs)):
        count = count + 1
        W0 = W0 -alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W0)       
        if count % monitor_period == 0 :
            weight.append(W0.copy())
    return weight        

# gradient descent with nesterov momentum
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
# monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" epochs
def gd_nesterov(Xs, Ys, gamma, W0, alpha, beta, num_epochs, monitor_period):
    # TODO students should implement this
    d, n = Xs.shape
    W = W0
    V0 = W0
    grad = []
    ii = np.arange(n)
    for t in tqdm(range(num_epochs)):
        V = W - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
        W = V + beta * (V - V0)
        V0 = V
        if ((t+1) % monitor_period == 0):
            grad.append(W)
    return grad

# SGD: run stochastic gradient descent with minibatching and sequential sampling order (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_minibatch_sequential_scan(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should use their implementation from programming assignment 2
    d, n = Xs.shape
    W = W0
    grad = []
    p = 0

    for t in tqdm(range(num_epochs)):
        for i in range(int(n/B)):
            interval = np.arange(i*B, (i+1)*B)
            ii = np.random.choice(interval,size = B, replace=False)
            W = W - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            p += 1
            if (p % monitor_period == 0):
                grad.append(W)

    return grad

# SGD + Momentum: add momentum to the previous algorithm
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, monitor_period):
    # TODO students should implement this
    d, n = Xs.shape
    W = W0
    V = W0
    p = 0
    grad = []
    for t in tqdm(range(num_epochs)):
        for i in range(int(n/B)):
            interval = np.arange(i*B, (i+1)*B)
            ii = np.random.choice(interval, size=B, replace = False)
            V = beta * V - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            W = W + V
            p+=1
            if (p % monitor_period == 0):
                grad.append(W)
    return grad
# Adam Optimizer
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# rho1            first moment decay rate ρ1
# rho2            second moment decay rate ρ2
# B               minibatch size
# eps             small factor used to prevent division by zero in update step
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def adam(Xs, Ys, gamma, W0, alpha, rho1, rho2, B, eps, num_epochs, monitor_period):
    # TODO students should implement this
    d, n = Xs.shape
    c = Ys.shape[0]
    result = []
    t = 0
    r = np.zeros((c, d))
    s = np.zeros((c, d))
    weight = W0
    for k in tqdm(range(num_epochs)):
        for i in range(int(n/B)):
            t = t+1

            Brange = np.arange(i * B, (i + 1) * B)
            ii = np.random.choice(Brange, size=B, replace=False)

            gradient = multinomial_logreg_grad_i(Xs, Ys, ii, gamma, weight)
            s = rho1 * s + (1 - rho1) * gradient
            r = rho2 * r + (1 - rho2) * (gradient*gradient)
            scorrect = s / (1 - np.power(rho1, t))
            rcorrect = r / (1 - np.power(rho2, t))

            weight =  weight - alpha * scorrect / np.sqrt(rcorrect + eps)

            if (t % monitor_period == 0):
                result.append(weight)

    return result

#%%
if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO add code to produce figures
    d,n = Xs_tr.shape
    c,_ = Ys_tr.shape
    W0 = np.zeros((c,d))
    gamma = 0.0001
    alpha = 1.0
    num_epochs = 100
    monitor_period = 1
    beta_1 = 0.9
    beta_2 = 0.99
#%%

    Xs = np.array([[.8, .3, .1, .8],
                       [.5, .8, .5, .4]])
    Ys = np.array([[1, 0, 0, 1],
                       [0, 1, 1, 0]])
    W = np.zeros((2, 2))
    models = gradient_descent(Xs, Ys, 0, W, 1.0, 5, 1)
    print(len(models))

#%%
   # record the value of the parameters every iteration
    grad_tr1 = gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, num_epochs, monitor_period)
    grad_tr2 = gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha, beta_1, num_epochs, monitor_period)
    grad_tr3 = gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha, beta_2, num_epochs, monitor_period)
     # resulting error and loss
    x1 = []
    l = len(grad_tr1)
    y1 = []
    yt1 = []
    yl1 = []
    for i in tqdm(range(l)):
        x = i + 1
        y1.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr1[i]))
        yt1.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr1[i]))
        yl1.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr1[i]))
        x1.append(x)
        
    x2 = []
    l = len(grad_tr2)
    y2 = []
    yt2 = []
    yl2 = []
    for i in tqdm(range(l)):
        x = i + 1
        y2.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr2[i]))
        yt2.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr2[i]))
        yl2.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr2[i]))        
        x2.append(x)
        
    x3 = []
    l = len(grad_tr3)
    y3 = []
    yt3 = []
    yl3 = []
    for i in tqdm(range(l)):
        x = i + 1
        y3.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr3[i]))
        yt3.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr3[i]))
        yl3.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr3[i]))
        x3.append(x)
#    plot
    plt.figure(1)
    l1, = plt.plot(x1, y1)
    l2, = plt.plot(x2, y2)
    l3, = plt.plot(x3, y3)

    plt.title('Training error')
    plt.legend(handles = [l1,l2,l3],labels = ['GD','N-0.9','N-0.99'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('1.png')
    plt.close()
    
    plt.figure(2)
    l5, = plt.plot(x1, yt1)
    l6, = plt.plot(x2, yt2)
    l7, = plt.plot(x3, yt3)
    
    plt.title('Test error')
    plt.legend(handles = [l5,l6,l7],labels = ['GD','N-0.9','N-0.99'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('2.png') 
    plt.close()
    
    plt.figure(3)
    l4, = plt.plot(x1, yl1)
    l8, = plt.plot(x2, yl2)
    l9, = plt.plot(x3, yl3)
    
    plt.title('Training loss')
    plt.legend(handles = [l4,l8,l9],labels = ['GD','N-0.9','N-0.99'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('3.png') 
    plt.close()
    #%%
#    %% measure run time
    time_m = np.zeros(2)
    for i in range(5):

        start = time.time()
        gradient1 = gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, num_epochs, monitor_period)
        time_m[0] = time_m[0] + (time.time() - start)
               
        start = time.time()
        gradient2 = gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha, beta_1, num_epochs, monitor_period)
        time_m[1] = time_m[1] + (time.time() - start)
              
    
    for i in range(2):
        time_m[i] = time_m[i]/5
        
    print(time_m)
    #%% diffrent parameters
    alpha_1 = 0.5
    alpha_2 = 0.1
    beta_3 = 0.95
    beta_4 = 0.6
    
    grad_tr4 = gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha_1, num_epochs, monitor_period)
    grad_tr5 = gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha_2, num_epochs, monitor_period)
    grad_tr6 = gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha, beta_3, num_epochs, monitor_period)
    grad_tr7 = gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha, beta_4, num_epochs, monitor_period)
    
    x4 = []
    l = len(grad_tr4)
    y4 = []
    yt4 = []
    yl4 = []
    for i in tqdm(range(l)):
        x = i + 1
        y4.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr4[i]))
        yt4.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr4[i]))
        yl4.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr4[i]))
        x4.append(x)
        
    x5 = []
    l = len(grad_tr5)
    y5 = []
    yt5 = []
    yl5 = []
    for i in tqdm(range(l)):
        x = i + 1
        y5.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr5[i]))
        yt5.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr5[i]))
        yl5.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr5[i]))        
        x5.append(x)
        
    x6 = []
    l = len(grad_tr6)
    y6 = []
    yt6 = []
    yl6 = []
    for i in tqdm(range(l)):
        x = i + 1
        y6.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr6[i]))
        yt6.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr6[i]))
        yl6.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr6[i]))
        x6.append(x)
        
    x7 = []
    l = len(grad_tr7)
    y7 = []
    yt7 = []
    yl7 = []
    for i in tqdm(range(l)):
        x = i + 1
        y7.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr7[i]))
        yt7.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr7[i]))
        yl7.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr7[i]))
        x7.append(x)
   #%%     
#    plot
    plt.figure(4)
    l1, = plt.plot(x1, y1)
    l10, = plt.plot(x4, y4)
    l11, = plt.plot(x5, y5)
    plt.title('Training error')
    plt.legend(handles = [l1,l10,l11],labels = ['alpha = 1','alpha = 0.5','alpha = 0.1'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('4.png')
    plt.close()
    
    plt.figure(5)
    l5, = plt.plot(x1, yt1)
    l14, = plt.plot(x4, yt4)
    l15, = plt.plot(x5, yt5)
    plt.title('Test error')
    plt.legend(handles = [l5,l14,l15],labels = ['alpha = 1','alpha = 0.5','alpha = 0.1'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('5.png') 
    plt.close()
    
    plt.figure(6)

    plt.title('Training loss')
    l4, = plt.plot(x1, yl1)  
    l18, = plt.plot(x4, yl4) 
    l19, = plt.plot(x5, yl5)
    plt.legend(handles = [l4,l18,l19],labels = ['alpha = 1','alpha = 0.5','alpha = 0.1'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('6.png') 
    plt.close()
    
    plt.figure(7)

    plt.title('Training error')
    l2, = plt.plot(x2, y2)
    l3, = plt.plot(x3, y3)
    l12, = plt.plot(x6, y6)   
    l13, = plt.plot(x7, y7)
    plt.legend(handles = [l2,l3,l12,l13],labels = ['beta = 0.9','beta = 0.99','beta = 0.95','beta = 0.6'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('7.png')
    plt.close()
    
    plt.figure(8)

    plt.title('Test error')
    l6, = plt.plot(x2, yt2)
    l7, = plt.plot(x3, yt3)
    l16, = plt.plot(x6, yt6)
    l17, = plt.plot(x7, yt7)
    plt.legend(handles = [l6,l7,l16,l17],labels = ['beta = 0.9','beta = 0.99','beta = 0.95','beta = 0.6'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('8.png') 
    plt.close()
    
    plt.figure(9)

    plt.title('Training loss')
    l8, = plt.plot(x2, yl2)
    l9, = plt.plot(x3, yl3)
    l20, = plt.plot(x6, yl6)
    l21, = plt.plot(x7, yl7) 
    plt.legend(handles = [l8,l9,l20,l21],labels = ['beta = 0.9','beta = 0.99','beta = 0.95','beta = 0.6'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('9.png') 
    plt.close()
    #%% more parameter
    grad_tr8 = gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha_1, beta_1, num_epochs, monitor_period)
    grad_tr9 = gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha_2, beta_1, num_epochs, monitor_period)
    
    x8 = []
    l = len(grad_tr8)
    y8 = []
    yt8 = []
    yl8 = []
    for i in tqdm(range(l)):
        x = i + 1
        y8.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr8[i]))
        yt8.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr8[i]))
        yl8.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr8[i]))
        x8.append(x)
        
    x9 = []
    l = len(grad_tr9)
    y9 = []
    yt9 = []
    yl9 = []
    for i in tqdm(range(l)):
        x = i + 1
        y9.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr9[i]))
        yt9.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr9[i]))
        yl9.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr9[i]))        
        x9.append(x)
        
    plt.figure(10)
    l2, = plt.plot(x2, y2)
    l22, = plt.plot(x8, y8)
    l23, = plt.plot(x9, y9)
    plt.title('Training error')
    plt.legend(handles = [l2,l22,l23],labels = ['alpha = 1','alpha = 0.5','alpha = 0.1'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('10.png')
    plt.close()
    
    plt.figure(11)
    l6, = plt.plot(x1, yt1)
    l24, = plt.plot(x8, yt8)
    l25, = plt.plot(x9, yt9)
    plt.title('Test error')
    plt.legend(handles = [l6,l24,l25],labels = ['alpha = 1','alpha = 0.5','alpha = 0.1'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('11.png') 
    plt.close()
    
    plt.figure(12)

    plt.title('Training loss')
    l8, = plt.plot(x1, yl1)  
    l26, = plt.plot(x8, yl8) 
    l27, = plt.plot(x9, yl9)
    plt.legend(handles = [l8,l26,l27],labels = ['alpha = 1','alpha = 0.5','alpha = 0.1'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('12.png') 
    plt.close()
     #%% Part 2
    gamma = 0.0001
    alpha = 0.2
    num_epochs = 10
    monitor_period = 10
    B = 600
    beta_1 = 0.9
    beta_2 = 0.99
    #%% record the value of the parameters every iteration
    grad_tr1 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, num_epochs, monitor_period)
    grad_tr2 = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta_1, B, num_epochs, monitor_period)
    grad_tr3 = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta_2, B, num_epochs, monitor_period)

    x1 = []
    l = len(grad_tr1)
    y1 = []
    yt1 = []
    yl1 = []
    for i in tqdm(range(l)):
        x = i + 1
        y1.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr1[i]))
        yt1.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr1[i]))
        yl1.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr1[i]))
        x1.append(x)
        
    x2 = []
    l = len(grad_tr2)
    y2 = []
    yt2 = []
    yl2 = []
    for i in tqdm(range(l)):
        x = i + 1
        y2.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr2[i]))
        yt2.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr2[i]))
        yl2.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr2[i]))
        x2.append(x)
        
    x3 = []
    l = len(grad_tr3)
    y3 = []
    yt3 = []
    yl3 = []
    for i in tqdm(range(l)):
        x = i + 1
        y3.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr3[i]))
        yt3.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr3[i]))
        yl3.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr3[i]))
        x3.append(x)

    plt.figure(13)
    l1, = plt.plot(x1, y1)
    l2, = plt.plot(x2, y2)
    l3, = plt.plot(x3, y3)

    plt.title('Training error')
    plt.legend(handles = [l1,l2,l3],labels = ['SGD','MomentSGD-0.9','MomentSGD-0.99'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('13.png')
    plt.close()
    
    plt.figure(14)
    l5, = plt.plot(x1, yt1)
    l6, = plt.plot(x2, yt2)
    l7, = plt.plot(x3, yt3)
    
    plt.title('Test error')
    plt.legend(handles = [l5,l6,l7],labels = ['SGD','MomentSGD-0.9','MomentSGD-0.99'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('14.png') 
    plt.close()
    
    plt.figure(15)
    l4, = plt.plot(x1, yl1)
    l8, = plt.plot(x2, yl2)
    l9, = plt.plot(x3, yl3)
    
    plt.title('Training loss')
    plt.legend(handles = [l4,l8,l9],labels = ['SGD','MomentSGD-0.9','MomentSGD-0.99'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('15.png') 
    plt.close()

    time_m = np.zeros(2)
    for i in range(5):

        start = time.time()
        gradient1 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, num_epochs, monitor_period)
        time_m[0] = time_m[0] + (time.time() - start)
               
        start = time.time()
        gradient2 = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta_1, B, num_epochs, monitor_period)
        time_m[1] = time_m[1] + (time.time() - start)
        
    
    for i in range(2):
        time_m[i] = time_m[i]/5
        
    print(time_m)
    #%% diffrent parameters
    alpha_1 = 0.5
    alpha_2 = 0.1
    beta_3 = 0.6
    beta_4 = 0.2
    
    grad_tr4 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha_1, B, num_epochs, monitor_period)
    grad_tr5 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha_2, B, num_epochs, monitor_period)
    grad_tr6 = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta_3, B, num_epochs, monitor_period)
    grad_tr7 = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta_4, B, num_epochs, monitor_period)
    
    x4 = []
    l = len(grad_tr4)
    y4 = []
    yt4 = []
    yl4 = []
    for i in tqdm(range(l)):
        x = i + 1
        y4.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr4[i]))
        yt4.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr4[i]))
        yl4.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr4[i]))
        x4.append(x)
        
    x5 = []
    l = len(grad_tr5)
    y5 = []
    yt5 = []
    yl5 = []
    for i in tqdm(range(l)):
        x = i + 1
        y5.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr5[i]))
        yt5.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr5[i]))
        yl5.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr5[i]))        
        x5.append(x)
        
    x6 = []
    l = len(grad_tr6)
    y6 = []
    yt6 = []
    yl6 = []
    for i in tqdm(range(l)):
        x = i + 1
        y6.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr6[i]))
        yt6.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr6[i]))
        yl6.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr6[i]))
        x6.append(x)
        
    x7 = []
    l = len(grad_tr7)
    y7 = []
    yt7 = []
    yl7 = []
    for i in tqdm(range(l)):
        x = i + 1
        y7.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr7[i]))
        yt7.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr7[i]))
        yl7.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr7[i]))
        x7.append(x)
   #%%     
#    plot
    plt.figure(16)
    l1, = plt.plot(x1, y1)
    l10, = plt.plot(x4, y4)
    l11, = plt.plot(x5, y5)
    plt.title('Training error')
    plt.legend(handles = [l1,l10,l11],labels = ['alpha = 1','alpha = 0.5','alpha = 0.1'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('16.png')
    plt.close()
    
    plt.figure(17)
    l5, = plt.plot(x1, yt1)
    l14, = plt.plot(x4, yt4)
    l15, = plt.plot(x5, yt5)
    plt.title('Test error')
    plt.legend(handles = [l5,l14,l15],labels = ['alpha = 1','alpha = 0.5','alpha = 0.1'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('17.png') 
    plt.close()
    
    plt.figure(18)

    plt.title('Training loss')
    l4, = plt.plot(x1, yl1)  
    l18, = plt.plot(x4, yl4) 
    l19, = plt.plot(x5, yl5)
    plt.legend(handles = [l4,l18,l19],labels = ['alpha = 1','alpha = 0.5','alpha = 0.1'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('18.png') 
    plt.close()
    #%%
    plt.figure(19)

    plt.title('Training error')
    l2, = plt.plot(x2, y2)
    l3, = plt.plot(x3, y3)
    l12, = plt.plot(x6, y6)   
    l13, = plt.plot(x7, y7)
    plt.legend(handles = [l2,l3,l12,l13],labels = ['beta = 0.9','beta = 0.99','beta = 0.6','beta = 0.2'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('19.png')
    plt.close()
    
    plt.figure(20)

    plt.title('Test error')
    l6, = plt.plot(x2, yt2)
    l7, = plt.plot(x3, yt3)
    l16, = plt.plot(x6, yt6)
    l17, = plt.plot(x7, yt7)
    plt.legend(handles = [l6,l7,l16,l17],labels = ['beta = 0.9','beta = 0.99','beta = 0.6','beta = 0.2'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('20.png') 
    plt.close()
    
    plt.figure(21)

    plt.title('Training loss')
    l8, = plt.plot(x2, yl2)
    l9, = plt.plot(x3, yl3)
    l20, = plt.plot(x6, yl6)
    l21, = plt.plot(x7, yl7) 
    plt.legend(handles = [l8,l9,l20,l21],labels = ['beta = 0.9','beta = 0.99','beta = 0.6','beta = 0.2'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('21.png') 
    plt.close()
    #%% more parameter
    grad_tr8 = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha_1, beta_1, B, num_epochs, monitor_period)
    grad_tr9 = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha_2, beta_1, B, num_epochs, monitor_period)
    
    x8 = []
    l = len(grad_tr8)
    y8 = []
    yt8 = []
    yl8 = []
    for i in tqdm(range(l)):
        x = i + 1
        y8.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr8[i]))
        yt8.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr8[i]))
        yl8.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr8[i]))
        x8.append(x)
        
    x9 = []
    l = len(grad_tr9)
    y9 = []
    yt9 = []
    yl9 = []
    for i in tqdm(range(l)):
        x = i + 1
        y9.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr9[i]))
        yt9.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr9[i]))
        yl9.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr9[i]))        
        x9.append(x)
        
    plt.figure(22)
    l2, = plt.plot(x2, y2)
    l22, = plt.plot(x8, y8)
    l23, = plt.plot(x9, y9)
    plt.title('Training error')
    plt.legend(handles = [l2,l22,l23],labels = ['alpha = 1','alpha = 0.5','alpha = 0.1'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('22.png')
    plt.close()
    
    plt.figure(23)
    l6, = plt.plot(x1, yt1)
    l24, = plt.plot(x8, yt8)
    l25, = plt.plot(x9, yt9)
    plt.title('Test error')
    plt.legend(handles = [l6,l24,l25],labels = ['alpha = 1','alpha = 0.5','alpha = 0.1'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('23.png') 
    plt.close()
    
    plt.figure(24)

    plt.title('Training loss')
    l8, = plt.plot(x1, yl1)  
    l26, = plt.plot(x8, yl8) 
    l27, = plt.plot(x9, yl9)
    plt.legend(handles = [l8,l26,l27],labels = ['alpha = 1','alpha = 0.5','alpha = 0.1'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('24.png') 
    plt.close()
     #%% Part 3
    gamma = 0.0001
    alpha = 0.2
    num_epochs = 10
    monitor_period = 10
    B = 600
    alpha_M = 0.01
    rho1 = 0.9
    rho2 = 0.999
    eps = np.power(0.1,5)
    #%% record the value of the parameters every iteration
    grad_tr1 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, num_epochs, monitor_period)
    grad_tr2 = adam(Xs_tr, Ys_tr, gamma, W0, alpha_M, rho1, rho2, B, eps, num_epochs, monitor_period)

    x1 = []
    l = len(grad_tr1)
    y1 = []
    yt1 = []
    yl1 = []
    for i in tqdm(range(l)):
        x = i + 1
        y1.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr1[i]))
        yt1.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr1[i]))
        yl1.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr3[i]))
        x1.append(x)
        
    x2 = []
    l = len(grad_tr2)
    y2 = []
    yt2 = []
    yl2 = []
    for i in tqdm(range(l)):
        x = i + 1
        y2.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr2[i]))
        yt2.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr2[i]))
        yl2.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr2[i]))
        x2.append(x)

    plt.figure(25)
    l1, = plt.plot(x1, y1)
    l2, = plt.plot(x2, y2)

    plt.title('Training error')
    plt.legend(handles = [l1,l2],labels = ['SGD','adam'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('25.png')
    plt.close()
    
    plt.figure(26)
    l5, = plt.plot(x1, yt1)
    l6, = plt.plot(x2, yt2)
    
    plt.title('Test error')
    plt.legend(handles = [l5,l6],labels = ['SGD','adam'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('26.png') 
    plt.close()
    
    plt.figure(27)
    l4, = plt.plot(x1, yl1)
    l8, = plt.plot(x2, yl2)
    
    plt.title('Training loss')
    plt.legend(handles = [l4,l8],labels = ['SGD','adam'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('27.png') 
    plt.close()

    time_m = np.zeros(2)
    #%%
    for i in range(5):

        start = time.time()
        gradient1 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, num_epochs, monitor_period)
        time_m[0] = time_m[0] + (time.time() - start)
               
        start = time.time()
        gradient2 = adam(Xs_tr, Ys_tr, gamma, W0, alpha, rho1, rho2, B, eps, num_epochs, monitor_period)
        time_m[1] = time_m[1] + (time.time() - start)
        
        
    
    for i in range(2):
        time_m[i] = time_m[i]/5
        
    print(time_m)
    #%% diffrent parameters
    alpha_1 = 0.5
    alpha_2 = 0.1
    rho1_1 = 0.8
    rho1_2 = 0.99
    
    grad_tr4 = adam(Xs_tr, Ys_tr, gamma, W0, alpha_1, rho1, rho2, B, eps, num_epochs, monitor_period)
    grad_tr5 = adam(Xs_tr, Ys_tr, gamma, W0, alpha_2, rho1, rho2, B, eps, num_epochs, monitor_period)
    grad_tr6 = adam(Xs_tr, Ys_tr, gamma, W0, alpha, rho1_1, rho2, B, eps, num_epochs, monitor_period)
    grad_tr7 = adam(Xs_tr, Ys_tr, gamma, W0, alpha, rho1_2, rho2, B, eps, num_epochs, monitor_period)
    
    x4 = []
    l = len(grad_tr4)
    y4 = []
    yt4 = []
    yl4 = []
    for i in tqdm(range(l)):
        x = i + 1
        y4.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr4[i]))
        yt4.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr4[i]))
        yl4.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr4[i]))
        x4.append(x)
        
    x5 = []
    l = len(grad_tr5)
    y5 = []
    yt5 = []
    yl5 = []
    for i in tqdm(range(l)):
        x = i + 1
        y5.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr5[i]))
        yt5.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr5[i]))
        yl5.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr5[i]))        
        x5.append(x)
        
    x6 = []
    l = len(grad_tr6)
    y6 = []
    yt6 = []
    yl6 = []
    for i in tqdm(range(l)):
        x = i + 1
        y6.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr6[i]))
        yt6.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr6[i]))
        yl6.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr6[i]))
        x6.append(x)
        
    x7 = []
    l = len(grad_tr7)
    y7 = []
    yt7 = []
    yl7 = []
    for i in tqdm(range(l)):
        x = i + 1
        y7.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr7[i]))
        yt7.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr7[i]))
        yl7.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr7[i]))
        x7.append(x)
   #%%     
#    plot
    plt.figure(28)
    l1, = plt.plot(x1, y1)
    l10, = plt.plot(x4, y4)
    l11, = plt.plot(x5, y5)
    plt.title('Training error')
    plt.legend(handles = [l1,l10,l11],labels = ['alpha = 1','alpha = 0.5','alpha = 0.1'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('28.png')
    plt.close()
    
    plt.figure(29)
    l5, = plt.plot(x1, yt1)
    l14, = plt.plot(x4, yt4)
    l15, = plt.plot(x5, yt5)
    plt.title('Test error')
    plt.legend(handles = [l5,l14,l15],labels = ['alpha = 1','alpha = 0.5','alpha = 0.1'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('29.png') 
    plt.close()
    
    plt.figure(30)

    plt.title('Training loss')
    l4, = plt.plot(x1, yl1)  
    l18, = plt.plot(x4, yl4) 
    l19, = plt.plot(x5, yl5)
    plt.legend(handles = [l4,l18,l19],labels = ['alpha = 1','alpha = 0.5','alpha = 0.1'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('30.png') 
    plt.close()
    
    plt.figure(31)

    plt.title('Training error')
    l2, = plt.plot(x2, y2)
    l12, = plt.plot(x6, y6)   
    l13, = plt.plot(x7, y7)
    plt.legend(handles = [l2,l12,l13],labels = ['rho1 = 0.9','rho1 = 0.8','rho1 = 0.99'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('31.png')
    plt.close()
    
    plt.figure(32)

    plt.title('Test error')
    l6, = plt.plot(x2, yt2)
    l16, = plt.plot(x6, yt6)
    l17, = plt.plot(x7, yt7)
    plt.legend(handles = [l6,l16,l17],labels = ['rho1 = 0.9','rho1 = 0.8','rho1 = 0.99'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('32.png') 
    plt.close()
    
    plt.figure(33)

    plt.title('Training loss')
    l8, = plt.plot(x2, yl2)
    l20, = plt.plot(x6, yl6)
    l21, = plt.plot(x7, yl7) 
    plt.legend(handles = [l8,l20,l21],labels = ['rho1 = 0.9','rho1 = 0.8','rho1 = 0.99'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('33.png') 
    plt.close()
    #%% more parameter
    rho2_1 = 0.8
    rho2_2 = 0.6
    grad_tr8 = adam(Xs_tr, Ys_tr, gamma, W0, alpha, rho1, rho2_1, B, eps, num_epochs, monitor_period)
    grad_tr9 = adam(Xs_tr, Ys_tr, gamma, W0, alpha, rho1, rho2_2, B, eps, num_epochs, monitor_period)
    
    x8 = []
    l = len(grad_tr8)
    y8 = []
    yt8 = []
    yl8 = []
    for i in tqdm(range(l)):
        x = i + 1
        y8.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr8[i]))
        yt8.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr8[i]))
        yl8.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr8[i]))
        x8.append(x)
        
    x9 = []
    l = len(grad_tr9)
    y9 = []
    yt9 = []
    yl9 = []
    for i in tqdm(range(l)):
        x = i + 1
        y9.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr9[i]))
        yt9.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr9[i]))
        yl9.append(multinomial_logreg_loss(Xs_tr, Ys_tr, gamma, grad_tr9[i]))        
        x9.append(x)
        
    plt.figure(34)
    l2, = plt.plot(x2, y2)
    l22, = plt.plot(x8, y8)
    l23, = plt.plot(x9, y9)
    plt.title('Training error')
    plt.legend(handles = [l2,l22,l23],labels = ['rho2 = 0.999','rho2 = 0.8','rho2 = 0.6'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('34.png')
    plt.close()
    
    plt.figure(35)
    l6, = plt.plot(x1, yt1)
    l24, = plt.plot(x8, yt8)
    l25, = plt.plot(x9, yt9)
    plt.title('Test error')
    plt.legend(handles = [l6,l24,l25],labels = ['rho2 = 0.999','rho2 = 0.8','rho2 = 0.6'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('35.png') 
    plt.close()
    
    plt.figure(36)

    plt.title('Training loss')
    l8, = plt.plot(x1, yl1)  
    l26, = plt.plot(x8, yl8) 
    l27, = plt.plot(x9, yl9)
    plt.legend(handles = [l8,l26,l27],labels = ['rho2 = 0.999','rho2 = 0.8','rho2 = 0.6'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('36.png') 
    plt.close()