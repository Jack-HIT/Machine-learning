#!/usr/bin/env python3
import os

# BEGIN THREAD SETTINGS this sets the number of threads used by numpy in the program (should be set to 1 for Parts 1 and 3)
implicit_num_threads = 8
os.environ["OMP_NUM_THREADS"] = str(implicit_num_threads)
os.environ["MKL_NUM_THREADS"] = str(implicit_num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(implicit_num_threads)
# END THREAD SETTINGS

import numpy
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot
import threading
import time

from tqdm import tqdm

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables
import matplotlib.pyplot as plt

# SOME UTILITY FUNCTIONS that you may find to be useful, from my PA3 implementation
# feel free to use your own implementation instead if you prefer
def multinomial_logreg_error(Xs, Ys, W):
    predictions = numpy.argmax(numpy.dot(W, Xs), axis=0)
    error = numpy.mean(predictions != numpy.argmax(Ys, axis=0))
    return error

def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    WdotX = numpy.dot(W, Xs[:,ii])
    expWdotX = numpy.exp(WdotX - numpy.amax(WdotX, axis=0))
    softmaxWdotX = expWdotX / numpy.sum(expWdotX, axis=0)
    return numpy.dot(softmaxWdotX - Ys[:,ii], Xs[:,ii].transpose()) / len(ii) + gamma * W
# END UTILITY FUNCTIONS


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = numpy.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        numpy.random.seed(4787)
        perm = numpy.random.permutation(60000)
        Xs_tr = numpy.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = numpy.ascontiguousarray(Ys_tr[:,perm])
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = numpy.ascontiguousarray(Xs_te)
        Ys_te = numpy.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset



# SGD + Momentum (adapt from Programming Assignment 3)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    # TODO students should use their implementation from programming assignment 3
    # or adapt this version, which is from my own solution to programming assignment 3
    #models = []
    (d, n) = Xs.shape
    V = numpy.zeros(W0.shape)
    W = W0
    print("Running minibatch sequential-scan SGD with momentum")
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            ii = range(ibatch*B, (ibatch+1)*B)
            V = beta * V - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            W = W + V
            # if ((ibatch+1) % monitor_period == 0):
            #     models.append(W)
    return W


# SGD + Momentum (No Allocation) => all operations in the inner loop should be a
#   call to a numpy.____ function with the "out=" argument explicitly specified
#   so that no extra allocations occur
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
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_noalloc(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO students should initialize the parameter vector W and pre-allocate any needed arrays here
    V = numpy.zeros(W0.shape)
    W = W0
    WX_alloc = numpy.zeros((c,B))
    numpy.ascontiguousarray(WX_alloc)
    SUM_alloc = numpy.zeros(B)
    numpy.ascontiguousarray(SUM_alloc )
    grad = numpy.zeros((c,d))
    numpy.ascontiguousarray(grad)
    # Xsamp = numpy.zeros((int(n/B),d,B))
    # numpy.ascontiguousarray(Xsamp)
    # Ysamp = numpy.zeros((int(n/B),c,B))
    # numpy.ascontiguousarray(Ysamp)
    Xsamp = []
    
    Ysamp = []
    
    for i in range(int(n/B)):
        ii = range(i*B, (i+1)*B)
        Xsamp.append(Xs[:,ii])
        Ysamp.append(Ys[:,ii])
    numpy.ascontiguousarray(Xsamp)
    numpy.ascontiguousarray(Ysamp)
    print("Running minibatch sequential-scan SGD with momentum (no allocation)")
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            # ii = range(ibatch*B, (ibatch+1)*B)
            # TODO this section of code should only use numpy operations with the "out=" argument specified (students should implement this)
            numpy.sum(numpy.exp(numpy.dot(W,Xsamp[ibatch],out=WX_alloc),out=WX_alloc),axis = 0,out=SUM_alloc)
            numpy.subtract(numpy.exp(numpy.dot(W,Xsamp[ibatch],out=WX_alloc),out=WX_alloc)/SUM_alloc,Ysamp[ibatch],out=WX_alloc)
            numpy.add((grad + numpy.dot(WX_alloc,Xsamp[ibatch].T,out=grad))/B , gamma*W,out=grad)
            V = beta*V-alpha*grad
            numpy.add(W,V,out=W)
    return W


# SGD + Momentum (threaded)
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
# num_threads     how many threads to use
#
# returns         the final model arrived at at the end of training

def sgd_mss_with_momentum_threaded(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, num_threads):
    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO perform any global setup/initialization/allocation (students should implement this)
    V = numpy.zeros(W0.shape)
    W = W0
    grad = numpy.zeros((c,d,num_threads))
    gradAvg = numpy.zeros((c,d))
    # construct the barrier object
    iter_barrier = threading.Barrier(num_threads + 1)
    # we need to get Bt, the size for B in different threads
    Bt = int(numpy.floor(B / num_threads))
    # a function for each thread to run
    def thread_main(ithread, weight, volume, gradient, avgGradient):
        # TODO perform any per-thread allocations
        WX_alloc = numpy.zeros((c,Bt))
        numpy.ascontiguousarray(WX_alloc)
        SUM_alloc = numpy.zeros(Bt)
        numpy.ascontiguousarray(SUM_alloc)
        Xsamp = numpy.zeros((int(n/B),d,Bt))
        numpy.ascontiguousarray(Xsamp)
        Ysamp = numpy.zeros((int(n/B),c,Bt))
        numpy.ascontiguousarray(Ysamp)
        for ibatch in range(int(n/B)):
            ii = range(ibatch*B + ithread*Bt, ibatch*B + (ithread+1)*Bt)
            Xsamp[ibatch,:,:] = Xs[:,ii]
            Ysamp[ibatch,:,:] = Ys[:,ii]
            
        for it in range(num_epochs):
            for ibatch in range(int(n/B)):
                # TODO work done by thread in each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
                SfSum = numpy.sum(numpy.exp(numpy.dot(W,Xsamp[ibatch,:,:],out=WX_alloc),out=WX_alloc),axis = 0,out=SUM_alloc)
                Sf = numpy.subtract(numpy.exp(numpy.dot(W,Xsamp[ibatch,:,:],out=WX_alloc),out=WX_alloc)/SfSum,Ysamp[ibatch,:,:],out=WX_alloc)
                gradient[:,:,ithread] = gradient[:,:,ithread] + numpy.dot(Sf,Xsamp[ibatch,:,:].T)
                iter_barrier.wait()
                if ithread == 0:
                    avgGradient = numpy.sum(gradient, axis=2, out=avgGradient) / B + gamma * weight
                    volume = beta*volume-alpha*avgGradient
                    weight = weight + volume
                iter_barrier.wait()
                
    # worker_threads = [threading.Thread(target=thread_main, args=(it,)) for it in range(num_threads)]
    worker_threads = [threading.Thread(target=thread_main, args=(it, W, V, grad, gradAvg)) for it in range(num_threads)] 
    for t in worker_threads:
        print("running thread ", t)
        t.start()

    print("Running minibatch sequential-scan SGD with momentum (%d threads)" % num_threads)
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            iter_barrier.wait()
            # TODO work done on a single thread at each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
            iter_barrier.wait()

    for t in worker_threads:
        t.join()

    # return the learned model
    return W


# SGD + Momentum (No Allocation) in 32-bits => all operations in the inner loop should be a
#   call to a numpy.____ function with the "out=" argument explicitly specified
#   so that no extra allocations occur
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
# returns         the final model arrived at at the end of training

def sgd_mss_with_momentum_noalloc_float32(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    Xs = Xs.astype(numpy.float32)
    Ys = Ys.astype(numpy.float32)
    W0 = W0.astype(numpy.float32)

    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO students should implement this by copying and adapting their 64-bit code
    V = numpy.zeros(W0.shape, dtype=numpy.float32)
    W = W0
    WX_alloc = numpy.zeros((c,B), dtype=numpy.float32)
    numpy.ascontiguousarray(WX_alloc)
    SUM_alloc = numpy.zeros(B, dtype=numpy.float32)
    numpy.ascontiguousarray(SUM_alloc )
    grad = numpy.zeros((c,d), dtype=numpy.float32)
    numpy.ascontiguousarray(grad)
    Xsamp = numpy.zeros((int(n/B),d,B), dtype=numpy.float32)
    numpy.ascontiguousarray(Xsamp)
    Ysamp = numpy.zeros((int(n/B),c,B), dtype=numpy.float32)
    numpy.ascontiguousarray(Ysamp)
    for i in range(int(n/B)):
        ii = range(i*B, (i+1)*B)
        Xsamp[i,:,:]= Xs[:,ii]
        Ysamp[i,:,:]= Ys[:,ii]
    print("Running minibatch sequential-scan SGD with momentum (no allocation)")
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            # ii = range(ibatch*B, (ibatch+1)*B)
            # TODO this section of code should only use numpy operations with the "out=" argument specified (students should implement this)
            numpy.sum(numpy.exp(numpy.dot(W,Xsamp[ibatch,:,:],out=WX_alloc),dtype=numpy.float32,out=WX_alloc),axis = 0,dtype=numpy.float32,out=SUM_alloc)
            numpy.subtract(numpy.exp(numpy.dot(W,Xsamp[ibatch,:,:],out=WX_alloc),dtype=numpy.float32,out=WX_alloc)/SUM_alloc,Ysamp[ibatch,:,:],dtype=numpy.float32,out=WX_alloc)
            numpy.add((grad + numpy.dot(WX_alloc,Xsamp[ibatch,:,:].T,out=grad))/B , gamma*W,dtype=numpy.float32,out=grad)
            V = beta*V-alpha*grad
            numpy.add(W,V,dtype=numpy.float32,out=W)
    return W


# SGD + Momentum (threaded, float32)
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
# num_threads     how many threads to use
#
# returns         the final model arrived at at the end of training

def sgd_mss_with_momentum_threaded_float32(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, num_threads):
    (d, n) = Xs.shape
    (c, d) = W0.shape
    Xs = Xs.astype(numpy.float32)
    Ys = Ys.astype(numpy.float32)
    W0 = W0.astype(numpy.float32)
    # TODO students should implement this by copying and adapting their 64-bit code
    V = numpy.zeros(W0.shape,dtype=numpy.float32)
    W = W0
    grad = numpy.zeros((c,d,num_threads),dtype=numpy.float32)
    gradAvg = numpy.zeros((c,d),dtype=numpy.float32)
    # construct the barrier object
    iter_barrier = threading.Barrier(num_threads + 1)
    # we need to get Bt, the size for B in different threads
    Bt = int(numpy.floor(B / num_threads))
    # a function for each thread to run
    def thread_main(ithread, weight, volume, gradient, avgGradient):
        # TODO perform any per-thread allocations
        WX_alloc = numpy.zeros((c,Bt),dtype=numpy.float32)
        numpy.ascontiguousarray(WX_alloc)
        SUM_alloc = numpy.zeros(Bt,dtype=numpy.float32)
        numpy.ascontiguousarray(SUM_alloc)
        Xsamp = numpy.zeros((int(n/B),d,Bt),dtype=numpy.float32)
        numpy.ascontiguousarray(Xsamp)
        Ysamp = numpy.zeros((int(n/B),c,Bt),dtype=numpy.float32)
        numpy.ascontiguousarray(Ysamp)
        for ibatch in range(int(n/B)):
            ii = range(ibatch*B + ithread*Bt, ibatch*B + (ithread+1)*Bt)
            Xsamp[ibatch,:,:] = Xs[:,ii]
            Ysamp[ibatch,:,:] = Ys[:,ii]
            
        for it in range(num_epochs):
            for ibatch in range(int(n/B)):
                # TODO work done by thread in each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
                SfSum = numpy.sum(numpy.exp(numpy.dot(W,Xsamp[ibatch,:,:],out=WX_alloc),dtype=numpy.float32,out=WX_alloc),axis = 0,dtype=numpy.float32,out=SUM_alloc)
                Sf = numpy.subtract(numpy.exp(numpy.dot(W,Xsamp[ibatch,:,:],out=WX_alloc),dtype=numpy.float32,out=WX_alloc)/SfSum,Ysamp[ibatch,:,:],dtype=numpy.float32,out=WX_alloc)
                gradient[:,:,ithread] = gradient[:,:,ithread] + numpy.dot(Sf,Xsamp[ibatch,:,:].T)
                iter_barrier.wait()
                if ithread == 0:
                    avgGradient = numpy.sum(gradient, axis=2, dtype=numpy.float32,out=avgGradient) / B + gamma * weight
                    volume = beta*volume-alpha*avgGradient
                    weight = weight + volume
                iter_barrier.wait()
                
    # worker_threads = [threading.Thread(target=thread_main, args=(it,)) for it in range(num_threads)]
    worker_threads = [threading.Thread(target=thread_main, args=(it, W, V, grad, gradAvg)) for it in range(num_threads)] 
    for t in worker_threads:
        print("running thread ", t)
        t.start()

    print("Running minibatch sequential-scan SGD with momentum (%d threads)" % num_threads)
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            iter_barrier.wait()
            # TODO work done on a single thread at each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
            iter_barrier.wait()

    for t in worker_threads:
        t.join()

    # return the learned model
    return W



if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    d,n = Xs_tr.shape
    c,n = Ys_tr.shape
    #%% Part 1
    # TODO add code to produce figures
    alpha = 0.1
    beta = 0.9
    B = 16
    gamma = 0.0001
    num_epochs = 20
    W0 = numpy.zeros((c,d))
    begin = time.time()
    W1 = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs)
    end = time.time()
    print(end-begin)
    begin = time.time()
    W2 = sgd_mss_with_momentum_noalloc(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs)
    end = time.time()
    print(end-begin)
    error = numpy.sum(W2-W1)
    print(error)
    #%% Part 1.2
    B = [8,16,30,60,200,600,3000]
    time1 = []
    time2 = []
    for i in range(len(B)):
        begin = time.time()
        W1 = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B[i], num_epochs)
        end = time.time()
        time1.append(end-begin)
        begin = time.time()
        W2 = sgd_mss_with_momentum_noalloc(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B[i], num_epochs)
        end = time.time()
        time2.append(end-begin)
        
    x = [1,2,3,4,5,6,7]
    plt.figure(1)
    plt.plot(x,time1,x, time2)
    plt.title('wall-clock times');plt.xlabel('B size: 8 ----------16---------- 30---------- 60-------- 200------ 600-------- 3000');plt.ylabel('time(s)');plt.legend(['SGD_momentum', 'SGD_prealloc']);plt.savefig('part1');plt.close()

    #%% Part 2
    time3 = []
    time4 = []
    
    for i in range(len(B)):
        begin = time.time()
        W1 = sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B[i], num_epochs)
        end = time.time()
        time3.append(end-begin)
        begin = time.time()
        W2 = sgd_mss_with_momentum_noalloc(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B[i], num_epochs)
        end = time.time()
        time4.append(end-begin)
        
    plt.figure(2)
    plt.plot(x, time1, x, time2, x, time3, x, time4)
    plt.title('wall-clock times');plt.xlabel('B size: 8 ----------16---------- 30---------- 60-------- 200------ 600-------- 3000');plt.ylabel('time(s)');plt.legend(['SGD_Mo thr=1', 'SGD_pre thr =1','SGD_Mo thr=4', 'SGD_pre thr=4']);plt.savefig('part2');plt.close()
    #%% Part 3
    B = [8,16,30,60,200,600,3000]
    time5 = []
    num_threads = 4
    for i in range(len(B)):
        begin = time.time()
        W1 = sgd_mss_with_momentum_threaded(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B[i], num_epochs, num_threads)
        end = time.time()
        time5.append(end-begin)
        
    x = [1,2,3,4,5,6,7]
    plt.figure(3)
    plt.plot(x, time1, x, time2, x, time3, x, time4, x, time5)
    plt.title('wall-clock times');plt.xlabel('B size: 8 ----------16---------- 30---------- 60-------- 200------ 600-------- 3000');plt.ylabel('time(s)');plt.legend(['SGD_Mo thr=1', 'SGD_pre thr =1','SGD_Mo thr=4', 'SGD_pre thr=4','SGD_multithr']);plt.savefig('part3');plt.close()
    #%% Part 4.2 1 Thr
    time6 = []
    for i in range(len(B)):
        begin = time.time()
        W1 = sgd_mss_with_momentum_noalloc_float32(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B[i], num_epochs)
        end = time.time()
        time6.append(end-begin)
#%%
    B = [8,16,30,60,200,600,3000]
    time7 = []
    for i in range(len(B)):
        begin = time.time()
        W1 = sgd_mss_with_momentum_threaded_float32(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B[i], num_epochs, num_threads)
        end = time.time()
        time7.append(end-begin)
        #%% multi
    time8 = []
    for i in range(len(B)):
        begin = time.time()
        W1 = sgd_mss_with_momentum_noalloc_float32(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B[i], num_epochs)
        end = time.time()
        time8.append(end-begin)
    #%% Fiure
    plt.figure(4)
    plt.plot(x,time1,x,time2,x,time3,x,time4,x,time5,x,time6,x,time7,x,time8)
    plt.title('wall-clock times');plt.xlabel('B size: 8 ----------16---------- 30---------- 60-------- 200------ 600-------- 3000');plt.ylabel('time(s)');plt.legend(['SGD_Mo thr=1', 'SGD_pre thr=1','SGD_Mo thr=4', 'SGD_pre thr=4','SGD_multithr','SGD_pre32 thr=1','SGD_multithr32','SGD_pre32 thr=4']);plt.savefig('part4');plt.close()
    #%% Fiure
    plt.figure(5)
    plt.plot(x,time2,x,time6,x,time8)
    plt.title('wall-clock times');plt.xlabel('B size: 8 ----------16---------- 30---------- 60-------- 200------ 600-------- 3000');plt.ylabel('time(s)');plt.legend([ 'SGD_pre thr=1','SGD_pre32 thr=1','SGD_pre32 thr=4']);plt.savefig('part4_3');plt.close()
        #%% Fiure
    plt.figure(6)
    plt.plot(x,time6,x,time7)
    plt.title('wall-clock times');plt.xlabel('B size: 8 ----------16---------- 30---------- 60-------- 200------ 600-------- 3000');plt.ylabel('time(s)');plt.legend(['SGD_pre32 thr=1','SGD_multithr32']);plt.savefig('part4_2');plt.close()