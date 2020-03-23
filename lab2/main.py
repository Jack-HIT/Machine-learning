# In[1]
import os
import numpy as np
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot as plt

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables
from tqdm import tqdm
from scipy.special import softmax
import time 
#import random
# In[2]
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
        Xs_tr = np.ascontiguousarray(Xs_tr)
        Ys_tr = np.ascontiguousarray(Ys_tr)
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


# compute the gradient of the multinomial logistic regression objective, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    # TODO students should implement this

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
# returns   the model error as a fraction of incorrect labels (a number between 0 and 1)
def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should use their implementation from programming assignment 1
    d, n = Xs.shape
    yTr = np.argmax(np.dot(W,Xs), axis=0)
    yLabel = np.argmax(Ys, axis=0)
    err = len(np.argwhere(yTr-yLabel))
    return err/n

# ALGORITHM 1: run stochastic gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of iterations (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" iterations
def stochastic_gradient_descent(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    # TODO students should implement this
    d, n = Xs.shape
    W = W0
    T = num_epochs * n
    grad = []
    grad.append(W)
    for t in tqdm(range(T)):
        ii = [random.randint(0,n)] # this is list, I make a mistake
        W = W - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
        if ((t+1) % monitor_period == 0):
            grad.append(W)
            
    return grad
# ALGORITHM 2: run stochastic gradient descent with sequential sampling order
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of iterations (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" iterations
def sgd_sequential_scan(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    # TODO students should implement this
    d, n = Xs.shape
    W = W0
    T = 0
    grad = []
    grad.append(W)
    for t in tqdm(range(num_epochs)):
        for i in range(n):
            T+=1
            ii = [i] # from i ton
            W = W - alpha*multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            if ((T+1) % monitor_period == 0):
                grad.append(W)
            
    return grad
# ALGORITHM 3: run stochastic gradient descent with minibatching
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
def sgd_minibatch(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    d, n = Xs.shape
    W = W0
    grad = []
    grad.append(W)
    p = 0
    T = int(num_epochs * n / B)

    for t in tqdm(range(T)):
        ii = [np.random.randint(0, n) for b in range(B)]
        W = W - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
        p += 1
        if (p % monitor_period == 0):
            grad.append(W)

    return grad

# ALGORITHM 4: run stochastic gradient descent with minibatching and sequential sampling order
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
    # TODO students should implement this
    d, n = Xs.shape
    W = W0
    grad = []
    grad.append(W)
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
# In[3] Part 1-3
if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO add code to produce figures
    d,n = Xs_tr.shape
    c,_ = Ys_tr.shape
    W0 = np.zeros((c,d))
    gamma = 0.0001
    alpha = 0.01
    epochs = 10
    freq = 6000
    #%%
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    
    Xs = np.array([[.8, .3, .1, .8],
                       [.5, .8, .5, .4]])
    Ys = np.array([[1, 0, 0, 1],
                       [0, 1, 1, 0]])
    W = np.zeros((2, 2))
    models =  stochastic_gradient_descent(Xs, Ys, 0, W, 1.0, 5, 2)
    print(type(models))
    print(len(models))
    print(np.testing.assert_allclose(models[ 1], [[ 0.203574, -0.273803],[-0.203574,  0.273803]], atol=0.0001))
    #%%
    grad_tr1 = stochastic_gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, epochs, freq)
    grad_tr2 = sgd_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, epochs, freq)
    #%%
    alpha = 0.05
    B = 60
    freq = 100
    grad_tr3 = sgd_minibatch(Xs_tr, Ys_tr, gamma, W0, alpha, B, epochs, freq)
    grad_tr4 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, epochs, freq)

    #%% plot
    x1 = []
    l = len(grad_tr1)
    y1 = []
    yt1 = []
    for i in tqdm(range(l)):
        x = i + 1
        y1.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr1[i]))
        yt1.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr1[i]))
        x1.append(x)
        
    x2 = []
    l = len(grad_tr2)
    y2 = []
    yt2 = []
    for i in tqdm(range(l)):
        x = i + 1
        y2.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr2[i]))
        yt2.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr2[i]))
        x2.append(x)
        
    x3 = []
    l = len(grad_tr3)
    y3 = []
    yt3 = []
    for i in tqdm(range(l)):
        x = i + 1
        y3.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr3[i]))
        yt3.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr3[i]))
        x3.append(x)
        
    x4 = []
    l = len(grad_tr4)
    y4 = []
    yt4 = []
    for i in tqdm(range(l)):
        x = i + 1
        y4.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr4[i]))
        yt4.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr4[i]))
        x4.append(x)
    #%% figure
    plt.figure(1)
    l1, = plt.plot(x1, y1)
    l2, = plt.plot(x2, y2)
    l3, = plt.plot(x3, y3)
    l4, = plt.plot(x4, y4)
    plt.title('Training error')
    plt.legend(handles = [l1,l2,l3,l4],labels = ['alg1','alg2','alg3','alg4'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('1.png')
    plt.close()
    
    plt.figure(2)
    l5, = plt.plot(x1, yt1)
    l6, = plt.plot(x2, yt2)
    l7, = plt.plot(x3, yt3)
    l8, = plt.plot(x4, yt4)
    plt.title('Test error')
    plt.legend(handles = [l5,l6,l7,l8],labels = ['alg1','alg2','alg3','alg4'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('2.png') 
    plt.close()
#%% Part2-1&2&3
    gamma = 0.0001
    alpha = 0.1
    epochs = 10
    freq = 6000
    grad_tr1 = stochastic_gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, epochs, freq)
    
    x_1 = []
    l = len(grad_tr1)
    y_0 = []
    y_t0 = []
    for i in tqdm(range(l)):
        x = i + 1
        y_0.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr1[i]))
        y_t0.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr1[i]))
        x_1.append(x)
        
    alpha = 0.01
    grad_tr1 = stochastic_gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, epochs, freq)
    l = len(grad_tr1)
    y_1 = []
    y_t1 = []
    for i in tqdm(range(l)):
        x = i + 1
        y_1.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr1[i]))
        y_t1.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr1[i]))
        
    alpha = 0.001
    grad_tr1 = stochastic_gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, epochs, freq)
    y_t2 = []
    y_2 = []
    for i in tqdm(range(l)):
        x = i + 1
        y_2.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr1[i]))
        y_t2.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr1[i]))
        
    alpha = 0.0001
    grad_tr1 = stochastic_gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, epochs, freq)
    y_t3 = []
    y_3 = []
    for i in tqdm(range(l)):
        x = i + 1
        y_3.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr1[i]))
        y_t3.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr1[i]))
        
    plt.figure(3)
    s0, = plt.plot(x_1, y_0)
    s1, = plt.plot(x_1, y_1)
    s2, = plt.plot(x_1, y_2)
    s3, = plt.plot(x_1, y_3)
    plt.title('Training error')
    plt.legend(handles = [s0,s1,s2,s3],labels = ['0.1','0.01','0.001','0.0001'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('3.png')
    plt.close()
    
    plt.figure(4)
    s4, = plt.plot(x_1, y_t0)
    s5, = plt.plot(x_1, y_t1)
    s6, = plt.plot(x_1, y_t2)
    s7, = plt.plot(x_1, y_t3)
    plt.title('Test error')
    plt.legend(handles = [s4,s5,s6,s7],labels = ['0.1','0.01','0.001','0.0001'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('4.png')
    plt.close()
    #%% Part 2-4
    alpha = 0.2
    B = 60
    freq = 100
    grad_tr4 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, epochs, freq)
    x_1 = []
    l = len(grad_tr4)
    y_1 = []
    y_t1 = []
    for i in tqdm(range(l)):
        x = i + 1
        y_1.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr4[i]))
        y_t1.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr4[i]))
        x_1.append(x)
        
    alpha = 0.1
    grad_tr4 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, epochs, freq)
    l = len(grad_tr4)
    y_t2 = []
    y_2 = []
    for i in tqdm(range(l)):
        x = i + 1
        y_2.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr4[i]))
        y_t2.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr4[i]))
        
    alpha = 0.01
    grad_tr4 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, epochs, freq)
    l = len(grad_tr4)
    y_t3 = []
    y_3 = []
    for i in tqdm(range(l)):
        x = i + 1
        y_3.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr4[i]))
        y_t3.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr4[i]))
        
    alpha = 0.001
    grad_tr4 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, epochs, freq)
    l = len(grad_tr4)
    y_t4 = []
    y_4 = []
    for i in tqdm(range(l)):
        x = i + 1
        y_4.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr4[i]))
        y_t4.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr4[i]))
        
    plt.figure(5)
    s1, = plt.plot(x_1, y_1)
    s2, = plt.plot(x_1, y_2)
    s3, = plt.plot(x_1, y_3)
    s4, = plt.plot(x_1, y_4)
    plt.title('Training error')
    plt.legend(handles = [s1,s2,s3,s4],labels = ['0.2','0.1','0.01','0.001'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('5.png') 
    plt.close()
    
    plt.figure(6)
    s5, = plt.plot(x_1, y_t1)
    s6, = plt.plot(x_1, y_t2)
    s7, = plt.plot(x_1, y_t3)
    s8, = plt.plot(x_1, y_t4)
    plt.title('Test error')
    plt.legend(handles = [s5,s6,s7,s8],labels = ['0.2','0.1','0.01','0.001'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('6.png') 
    plt.close()
    #%% batch size
    alpha = 0.05
    B = 6
    freq = 100
    grad_tr4 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, epochs, freq)
    l = len(grad_tr4)
    x_1 = []
    y_1 = []
    y_t1 = []
    for i in tqdm(range(l)):
        x = i + 1
        y_1.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr4[i]))
        y_t1.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr4[i]))
        x_1.append(x)
        
    B = 60
    grad_tr4 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, epochs, freq)
    l = len(grad_tr4)
    y_t2 = []
    y_2 = []
    x_2 = []
    for i in tqdm(range(l)):
        x = i + 1
        y_2.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr4[i]))
        y_t2.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr4[i]))
        x_2.append(x)
        
    B = 600
    grad_tr4 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, epochs, freq)
    l = len(grad_tr4)
    y_t3 = []
    y_3 = []
    x_3 = []
    for i in tqdm(range(l)):
        x = i + 1
        y_3.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr4[i]))
        y_t3.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr4[i]))
        x_3.append(x)
        
    B = 12
    grad_tr4 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, epochs, freq)
    l = len(grad_tr4)
    y_t4 = []
    y_4 = []
    x_4 = []
    for i in tqdm(range(l)):
        x = i + 1
        y_4.append(multinomial_logreg_error(Xs_tr, Ys_tr, grad_tr4[i]))
        y_t4.append(multinomial_logreg_error(Xs_te, Ys_te, grad_tr4[i]))
        x_4.append(x)
        
    plt.figure(7)
    s1, = plt.plot(x_1, y_1)
    s2, = plt.plot(x_2, y_2)
    s3, = plt.plot(x_3, y_3)
    s4, = plt.plot(x_4, y_4)
    plt.title('Training error')
    plt.legend(handles = [s1,s2,s3,s4],labels = ['6','60','600','12'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('7.png') 
    
    plt.figure(8)
    s5, = plt.plot(x_1, y_t1)
    s6, = plt.plot(x_2, y_t2)
    s7, = plt.plot(x_3, y_t3)
    s8, = plt.plot(x_4, y_t4)
    plt.title('Test error')
    plt.legend(handles = [s5,s6,s7,s8],labels = ['6','60','600','12'],loc = 'best')
    plt.xlabel('epochs')
    plt.savefig('8.png') 
    
    #%% time
    time_m = np.zeros(4)
    for i in range(5):
        gamma = 0.0001
        alpha = 0.01
        epochs = 10
        freq = 6000
        start = time.time()
        gradient1 = stochastic_gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, epochs, freq)
        time_m[0] = time_m[0] + (time.time() - start)
               
        start = time.time()
        gradient2 = sgd_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, epochs, freq)
        time_m[1] = time_m[1] + (time.time() - start)
        
        alpha = 0.05
        B = 60
        freq = 100
        start = time.time()
        gradient3 = sgd_minibatch(Xs_tr, Ys_tr, gamma, W0, alpha, B, epochs, freq)
        time_m[2] = time_m[2] +(time.time() - start)
        
        start = time.time()
        gradient4 = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, epochs, freq)
        time_m[3] = time_m [3] +(time.time() - start)
    
    for i in range(4):
        time_m[i] = time_m[i]/5
        
    print(time_m)
    
    