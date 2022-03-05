import tensorflow as tf
from tensorflow.keras import layers,models,optimizers



class simple_agent():

    def __init__(self,input_shape,layers):
        self.input_shape = input_shape
        self.layers = layers

        self.model = self.create_model()

    def create_model(self):
        #input the state of the environment 
        input_layer = layers.Input(shape=self.input_shape)

        layer = input_layer
        for l in self.layers:
            layer = layers.Dense(l,activation="relu")(layer)
        
        continuous_action = layers.Dense(5, activation="tanh")(layer)
        binary_action = layers.Dense(3, activation="sigmoid")(layer)
        
        return models.Model(inputs=input_layer, outputs=[continuous_action,binary_action])

    def predict(self,inputs):
        q_values = self.model(inputs)

    def fit_batch(self):
        pass
