import numpy as np



def sigmoid(x, deriv=False):
    #print x
    if(deriv==True):
        return sigmoid(x)*(1-sigmoid(x))
    return 1.0/(1.0+np.exp(-x))

#X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
#y = np.array([[0],[1],[1],[0]])

def fit(X, y):
    np.random.seed(0)
    syn0 = np.random.random((18,20)) - 1
    syn1 = np.random.random((20,1)) - 1
    #print X.shape
    for j in range(1):
        l0 = X
        l1 = sigmoid(np.dot(l0, syn0))
        l2 = sigmoid(np.dot(l1, syn1))
        l2_error = y - l2
        
        l2_delta = l2_error*sigmoid(l2, deriv=True)
        l1_error = l2_delta.dot(syn1.T)
        #print sigmoid(l1, deriv=True).shape
        l1_delta = l1_error * sigmoid(l1, deriv=True)

        print l2_delta.shape
        syn1 += 0.01*l1.T.dot(l2_delta)
        syn0 += 0.01*l0.T.dot(l1_delta)

    return syn0, syn1

def predict(X, syn0, syn1):
    l1 = sigmoid(np.dot(X, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    #print l2
    return np.argmax(l2)
