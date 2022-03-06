import tensorflow as tf
from tensorflow.keras import layers,models,optimizers,losses



class simple_dqn_agent():

    def __init__(self,input_shape,layers,learning_rate,discount):
        self.input_shape = input_shape
        self.layers = layers
        self.discount = discount

        self.binary_loss = losses.CategoricalCrossentropy()
        self.cont_loss = losses.MeanSquaredError()
        self.opt = optimizers.Adam(learning_rate=learning_rate)

        self.action_model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        #input the state of the environment 
        input_layer = layers.Input(shape=self.input_shape)

        layer = input_layer
        for l in self.layers:
            layer = layers.Dense(l,activation="relu")(layer)
        
        continuous_action = layers.Dense(5, activation="tanh")(layer)
        binary_action = layers.Dense(3, activation="sigmoid")(layer)
        
        return models.Model(inputs=input_layer, outputs=[continuous_action,binary_action])

    def sync(self):
        """copies the weights of the action model to the target model.
        Do this to transfer the learnings of the action model to the target.
        """
        self.target_model.set_weights(self.action_model.get_weights())

    def fit_batch(self,old_states,new_states,rewards):

        updated_continuous_q_values, updated_binary_q_vaules = self.target_model(new_states)
        updated_continuous_q_values = rewards + self.discount * tf.reduce_max(updated_continuous_q_values, axis=1)
        updated_binary_q_vaules = rewards + self.discount * tf.reduce_max(updated_binary_q_vaules, axis=1)


        with tf.GradientTape() as tape:
            continuous_q_values, binary_q_vaules = self.action_model(old_states)
            continuous_q_actions = tf.reduce_sum(continuous_q_values, axis=1)

            cont_loss = tf.reduce_mean(self.cont_loss(updated_continuous_q_values,continuous_q_values))
            binary_loss = tf.reduce_mean(self.binary_loss(continuous_q_actions,binary_q_vaules))
            loss = cont_loss + binary_loss

        grads = tape.gradient(loss, self.action_model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.action_model.trainable_variables))



