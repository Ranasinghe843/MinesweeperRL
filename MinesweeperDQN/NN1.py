from keras.models import Model, Sequential
#from keras.layers import Convolution2D, Flatten, Dense, Input
from keras import backend as K
import keras.callbacks
from keras.layers import Dense
import numpy as np

class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.losses = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        #self.acc.append(logs.get('acc'))

# Neural Network Design
# Step 2 in plan
# Number of layers: 4
#     - Input Layer: input dimension is 18 for state and action representation
#     - Hidden Layers:
#         - Dense Layer 1: 84 neurons, ReLU activation
#         - Dense Layer 2: 256 neurons, ReLU activation
#         - Dense Layer 3: 128 neurons, ReLU activation
#     - Output Layer: 81 neurons, Sigmoid activation (outputs Q-value)
# Training Configuration:
#     - Optimizer: Adam optimizer with a learning rate of 0.0001
#     - Loss Function: MSE
#     - Batch Size: 4x4 grid
#     - Discount Factor: γ = 0.99
#     - Epochs per Training: 1 epoch per fit call using model.fit()
# Output Metrics:
#     - Accuracy: measure of percentage of correct predictions
#     - History Tracking: self.hist tracks training loss over time
class NeuralNet:
    def __init__(self, bsize = (9, 9), gamma = 0.99):
        self.model = Sequential()
        self.model.add(Dense(81, activation='relu', input_dim=81))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(81, activation='sigmoid'))
        opt = keras.optimizers.Adam(learning_rate=0.0001)
        self.model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
        self.model.summary()
        #self.tb = keras.callbacks.TensorBoard(log_dir='/tmp/NN/logs', write_graph=True)
        self.bsize = bsize
        self.gamma = gamma
        self.hist = LossHistory()
    
    def forward(self, x):
        # Forward pass: pass the input `x` through each layer
        layer1_output = self.model.layers[0](x)  # First layer (input to first hidden)
        layer2_output = self.model.layers[1](layer1_output)  # Second layer (hidden to hidden)
        layer3_output = self.model.layers[2](layer2_output)  # Third layer (hidden to hidden)
        output = self.model.layers[3](layer3_output)  # Output layer (hidden to output)

        return output

    def predict(self, state):
        q = self.model.predict(state,verbose=0)
        return q

    def fit(self, x, y):
        #self.model.train_on_batch(x, y)
        self.model.fit(x, y, epochs=1, callbacks=[self.hist],verbose=0)     #, callbacks=[self.tb])


    def save(self, fpath = 'model.hdf'):
        self.model.save_weights(fpath)

    def load(self, fpath = 'model.hdf'):
        self.model.load_weights(fpath, by_name = False)