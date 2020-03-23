import os
import numpy as np
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot as plt

mnist_data_directory = os.path.join(os.path.dirname('d:\release'), "data")

# additional imports you may find useful for this assignment
from tqdm import tqdm
from scipy.special import softmax

# TODO add any additional imports and global variables
import time 
import random

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
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label

        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the cross-entropy loss of the classifier
#
# x         examples          (d)
# y         labels            (c)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss_i(x, y, gamma, W):
    # TODO students should implement this in Part 1
    x = (np.mat(x)).T
    y = (np.mat(y)).T  # c*d time d*1 = c*1, h is y_bar      
    R = (-1)*np.dot(y.T, (np.log(softmax(np.dot(W,x), axis=0)))) + gamma/2 * np.sum(W*W)  
    return float(R)

# compute the gradient of a single example of the multinomial logistic regression objective, with regularization
#
# x         training example   (d)
# y         training label     (c)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the gradient of the model parameters
def multinomial_logreg_grad_i(x, y, gamma, W):
    # TODO students should implement this in Part 1
    x = (np.mat(x)).T
    y = (np.mat(y)).T   
    L_grad = np.dot(softmax(np.dot(W,x),axis=0)-y, x.T)                            
    L = L_grad + gamma*W
    return L
           
# test that the function multinomial_logreg_grad_i is indeed the gradient of multinomial_logreg_loss_i
def test_gradient(x,y,v,alpha,gamma,W):
    # TODO students should implement this in Part 1
    f_g = multinomial_logreg_grad_i(x, y, gamma, W)
    f_1 = multinomial_logreg_loss_i(x + alpha *v, y, gamma, W)
    f_2 = multinomial_logreg_loss_i(x, y, gamma, W)
    temp = np.sum(np.dot(f_g,v.T))##question
    temp2 = (f_1 - f_2)/alpha
    mark = temp-temp2
    if (abs(mark) < 0.00001):
        print('Jingwen_method is correct!')
    else:
        print('Jingwen_method has problem')
    return None
    
# compute the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should implement this
    d, n = Xs.shape
    yTr = np.argmax(np.dot(W,Xs), axis=0)
    yLabel = np.argmax(Ys, axis=0)
    err = len(np.argwhere(yTr-yLabel))
    return err/n

# compute the gradient of the multinomial logistic regression objective, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the gradient of the model parameters
def multinomial_logreg_total_grad(Xs, Ys, gamma, W):
    # TODO students should implement this
    # a starter solution using an average of the example gradients
    (d,n) = Xs.shape
    acc = W * 0.0
#    for i in range(n):
#         acc += multinomial_logreg_grad_i(Xs[:,i], Ys[:,i], gamma, W)
    acc = (1.0/n) * np.dot(softmax(np.dot(W,Xs), axis=0)-Ys, Xs.T) + gamma*W     
    return acc

# compute the cross-entropy loss of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_total_loss(Xs, Ys, gamma, W):
    # TODO students should implement this
    # a starter solution using an average of the example gradients
    (d,n) = Xs.shape
    acc = 0.0
#    for i in range(n):
#        acc += multinomial_logreg_loss_i(Xs[:,i], Ys[:,i], gamma, W)
    acc = (-1.0/n)*np.sum(np.multiply(Ys, (np.log(softmax(np.dot(W,Xs), axis=0))))) + gamma/2*np.sum(W*W)
    return acc

# run gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs            training examples (d * n)
# Ys            training labels   (d * c)
# gamma         L2 regularization constant
# W0            the initial value of the parameters (c * d)
# alpha         step size/learning rate
# num_iters     number of iterations to run
# monitor_freq  how frequently to output the parameter vector
#
# returns       a list of models parameters, one every "monitor_freq" iterations
#               should return model parameters before iteration 0, iteration monitor_freq, iteration 2*monitor_freq, and again at the end
#               for a total of (num_iters/monitor_freq)+1 models, if num_iters is divisible by monitor_freq.
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_iters, monitor_freq):
    # TODO students should implement this
    count = 0
    weight = []
    weight.append(W0.copy())
    begin_time = time.time()
    for i in tqdm(range(num_iters)):
        count = count + 1
        W0 = W0 -alpha * multinomial_logreg_total_grad(Xs, Ys, gamma, W0)        
        if count % monitor_freq == 0 :
            weight.append(W0.copy())
    end_time = time.time()
    run_time = end_time - begin_time
    print('Time cost is: ', run_time)
    return weight        

# estimate the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
# nsamples  number of samples to use for the estimation
#
# returns   the estimated model error when sampling with replacement
def estimate_multinomial_logreg_error(Xs, Ys, W, nsamples):
    # TODO students should implement this
    (d,n) = Xs.shape
    num1 = random.randint(0,n - nsamples)
    Xs_s = Xs[:,num1: num1 + nsamples]
    Ys_s = Ys[:,num1: num1 + nsamples] 
    yTr = np.argmax(np.dot(W,Xs_s), axis=0)
    yLabel = np.argmax(Ys_s, axis=0)
    err = len(np.argwhere(yTr-yLabel))
    return err/nsamples

    #%%
if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    #%%    # P1
    # TODO add code to produce figures
    d, n = Xs_tr.shape
    c, _ = Ys_tr.shape
    W = np.zeros((c,d))
 #   W = np.random.randn(c, d) 
    gamma = 0.0001
    alpha = 0.00001
    
    test_gradient(Xs_tr[:,1],Ys_tr[:,1],Xs_tr[:,1],alpha,gamma,W)
    #%%# P2 & P3    
#    W0 = np.random.randn(c, d)
    W0 = np.zeros((c,d))
    num_iters = 10
    monitor_freq = 10
    gamma = 0.0001
    alpha = 1.0
    
    P2_result = gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, num_iters, monitor_freq)
    print('total of results: %d', np.shape(P2_result))
    #%%     P4
    num_iters = 1000
    P4_result = gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, num_iters, monitor_freq)
    #%%  plot
    ii = np.zeros(101)
    errTr = np.zeros(101)
    errTe = np.zeros(101)
    lossTr = np.zeros(101)
    lossTe = np.zeros(101)
    T_start = time.time()
    for i in tqdm(range(101)):
        errTr[i] = multinomial_logreg_error(Xs_tr, Ys_tr, P4_result[i])
        errTe[i] = multinomial_logreg_error(Xs_te, Ys_te, P4_result[i])
        lossTr[i] = multinomial_logreg_total_loss(Xs_tr, Ys_tr, gamma, P4_result[i])
        lossTe[i] = multinomial_logreg_total_loss(Xs_te, Ys_te, gamma, P4_result[i])
        ii[i] = i + 1
    T_end = time.time()
    T_run = T_end - T_start
    print('Time cost is :', T_run)
    #%%   Plot loss and error vs iterations 
    plt.subplot(2,2,1)
    plt.plot(ii, errTr)
    plt.title('Training error')
    plt.xlabel('iteration')
    plt.subplot(2,2,2)
    plt.plot(ii, errTe)
    plt.title('Test error')
    plt.xlabel('iteration')    
    plt.subplot(2,2,3)
    plt.title('Training loss')
    plt.xlabel('iteration')
    plt.plot(ii, lossTr)
    plt.subplot(2,2,4)
    plt.title('Test loss')
    plt.xlabel('iteration')
    plt.plot(ii, lossTe)
    plt.savefig('1.jpg') 
    #%%    subsample
    nsamples = 1000
    errTr_s = np.zeros(101)
    errTe_s = np.zeros(101)
    T_start1 = time.time()
    for i in tqdm(range(101)):
        errTr_s[i] = estimate_multinomial_logreg_error(Xs_tr, Ys_tr, P4_result[i], nsamples)
        errTe_s[i] = estimate_multinomial_logreg_error(Xs_te, Ys_te, P4_result[i], nsamples)
    T_end1 = time.time()
    T_run1 = T_end1 - T_start1
    print('Time cost is %f', T_run1)
    #%% subsample plot        
    plt.subplot(2,1,1)
    plt.plot(ii, errTr_s)
    plt.title('Training error')
    plt.xlabel('iteration')
    plt.subplot(2,1,2)
    plt.plot(ii, errTe_s)
    plt.title('Test error')
    plt.xlabel('iteration')     
    plt.savefig('2.jpg') 