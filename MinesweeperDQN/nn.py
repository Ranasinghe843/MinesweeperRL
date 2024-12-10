from scipy.misc import imresize
import numpy as np
from mlearn.neural_network import MLPClassifier

def resize_data(img):
	M=imresize(img,(8,8),interp='bicubic').flatten()
	bias=np.array([1])
        for i in range(64):
                if M[i] != 0:
                        M[i] = 1
	return M

file = open('optdigits-orig.tra', 'r')

count = 0
img = 0
X, Y = [], []
img = []
digits = [4, 7, 8]
mapping = {4 : '100', 7 : '010', 8 : '001'}
for i in range(21):
        file.readline()

for line in file:
        if count == 33:
                count = 0
                if y in digits:
                        Y.append(y)
                        X.append(resize_data(np.array(img).astype(np.float)).tolist())
                img = []
        if count == 32:
                y = float(line.strip())
        else:
                l = list(line.rstrip())
                img.append(l)
        count += 1
if y in digits:
        Y.append(y)
        X.append(resize_data(np.array(img).astype(np.float)).tolist())

X = np.array(X)
Y = np.array(Y)
y = []
for e in Y:
      y.append(list(mapping[e]))
y = np.array(y).astype(np.float)

#for i in range(X.shape[0]):
        #for j in range(X.shape[1]):
        #        if X[i][j] != 255:
        #                X[i][j] = 1

clf = MLPClassifier(hidden_layer_sizes=[20], max_iter=500, epsilon=0.01)
clf.fit(X, y)


#test
file = open('optdigits-orig.windep', 'r')
X, Y = [], []
count = 0
img = []
y = 0
correct = 0
for i in range(21):
        file.readline()

for line in file:
        if count == 33:
                count = 0
                if y in digits:
                        Y.append(y)
                        X.append(resize_data(np.array(img).astype(np.float)).tolist())
                img = []
        if count == 32:
                y = float(line.strip())
        else:
                l = list(line.rstrip())
                img.append(l)
        count += 1
if y in digits:
        Y.append(y)
        X.append(resize_data(np.array(img).astype(np.float)).tolist())

X = np.array(X)
print len(X)
for i in range(len(X)):
        val = clf.predict(X[i])
        if val == 0:
                predicted = 4
        elif val == 1:
                predicted = 7
        elif val == 2:
                predicted = 8

        print "Original:",
        print int(Y[i]),
	print " Predicted:",
	print predicted
	if predicted == int(Y[i]):
		correct += 1

print "Total test samples:", len(X)
print "Correctly predicted:", correct
print "Accuracy:", float(correct)*100/len(X)
