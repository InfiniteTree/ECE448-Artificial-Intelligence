import numpy as np
iteration_MAX_NUM = 50

def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    # TODO
    X = np.mat(X)
    # Using Gradient Descent method (GD) to get w
    for n in range(1,iteration_MAX_NUM+1): 
        for i in range(N):
            x = X[:,i].transpose() # For Matrix operations
            x = np.c_[x, np.array([1])]
            prob = sigmoid(np.dot(x,w)) # Prediction
            grads = np.array((y[:,i] - prob)) # gradients
            lr = 1/n # learning rate
            w += np.multiply(np.multiply(np.multiply((lr)*grads, prob), 1-prob),x.transpose())  
    
    # Reconstructe the elements' order of w for plotting
    temp = w[2]
    w = np.delete(w,2)
    w = np.r_[temp, w]
     # w0:Intercept on the y-axis; w1, w2 The slope of the decision boundary on the first/second feature axis in feature space
    w = np.array([[w[0]],[w[1]],[w[2]]])
    # print("Now w is",w)
    # end answer
    return w

