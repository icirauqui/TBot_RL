import numpy as np
from collections import deque
from numpy import linalg as LA

from tradeEnv import TradeEnv



# Global variables
NUM_EPISODES = 10000
MAX_TIMESTEPS = 1000
AVERAGE_REWARD_TO_SOLVE = 195
NUM_EPS_TO_SOLVE = 100
NUM_RUNS = 20
GAMMA = 0.95
EPSILON_DECAY = 0.997
update_size = 10
hidden_layer_size = 24
num_hidden_layers = 2
scores_last_timesteps = deque([], NUM_EPS_TO_SOLVE)

def relu(mat):
    return np.multiply(mat,(mat>0))
    
def relu_derivative(mat):
    return (mat>0)*1

class NNLayer:
    # class representing a neural net layer
    def __init__(self, input_size, output_size, activation=None, lr = 0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(input_size, output_size))
        self.stored_weights = np.copy(self.weights)
        self.activation_function = activation
        self.lr = lr
        self.m = np.zeros((input_size, output_size))
        self.v = np.zeros((input_size, output_size))
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.time = 1
        self.adam_epsilon = 0.00000001

    # Compute the forward pass for this layer
    def forward(self, inputs, remember_for_backprop=True):
        # inputs has shape batch_size x layer_input_size 
        input_with_bias = np.append(inputs,1)
        unactivated = None
        if remember_for_backprop:
            unactivated = np.dot(input_with_bias, self.weights)
        else: 
            unactivated = np.dot(input_with_bias, self.stored_weights)
        # store variables for backward pass
        output = unactivated
        if self.activation_function != None:
            # assuming here the activation function is relu, this can be made more robust
            output = self.activation_function(output)
        if remember_for_backprop:
            self.backward_store_in = input_with_bias
            self.backward_store_out = np.copy(unactivated)
        return output    
        
    def update_weights(self, gradient):        
        m_temp = np.copy(self.m)
        v_temp = np.copy(self.v) 
        
        m_temp = self.beta_1*m_temp + (1-self.beta_1)*gradient
        v_temp = self.beta_2*v_temp + (1-self.beta_2)*(gradient*gradient)
        m_vec_hat = m_temp/(1-np.power(self.beta_1, self.time+0.1))
        v_vec_hat = v_temp/(1-np.power(self.beta_2, self.time+0.1))
        self.weights = self.weights - np.divide(self.lr*m_vec_hat, np.sqrt(v_vec_hat)+self.adam_epsilon)
        
        self.m = np.copy(m_temp)
        self.v = np.copy(v_temp)
        
    def update_stored_weights(self):
        self.stored_weights = np.copy(self.weights)
        
    def update_time(self):
        self.time = self.time+1
        
    def backward(self, gradient_from_above):
        adjusted_mul = gradient_from_above
        # this is pointwise
        if self.activation_function != None:
            adjusted_mul = np.multiply(relu_derivative(self.backward_store_out),gradient_from_above)
        D_i = np.dot(np.transpose(np.reshape(self.backward_store_in, (1, len(self.backward_store_in)))), np.reshape(adjusted_mul, (1,len(adjusted_mul))))
        delta_i = np.dot(adjusted_mul, np.transpose(self.weights))[:-1]
        self.update_weights(D_i)
        return delta_i
        
class rlAgent:
    # class representing a reinforcement learning agent
    def __init__(self, df, num_hidden_layers=2, hidden_size=24, gamma=0.95, epsilon_decay=0.997, epsilon_min=0.01, learning_rate1=0.001):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque([],1000000)
        self.gamma = gamma

        self.env = TradeEnv(df)
        self.input_size = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        self.output_size = self.env.action_space.n
        print("Input", self.input_size, "   Output",self.output_size)
        
        self.layers = [NNLayer(self.input_size + 1, self.hidden_size, activation=relu,lr=learning_rate1)]
        for i in range(self.num_hidden_layers-1):
            self.layers.append(NNLayer(self.hidden_size+1, self.hidden_size, activation=relu,lr=learning_rate1))
        self.layers.append(NNLayer(self.hidden_size+1, self.output_size, activation=None,lr=learning_rate1))
        
    def select_action(self, observation):
        values = self.forward(np.asmatrix(observation))
        if (np.random.random() > self.epsilon):
            return np.argmax(values)
        else:
            return np.random.randint(self.env.action_space.n)
            
    def forward(self, observation, remember_for_backprop=True):
        vals = np.copy(observation)
        index = 0
        for layer in self.layers:
            vals = layer.forward(vals, remember_for_backprop)
            index = index + 1
        return vals
        
    def remember(self, done, action, observation, prev_obs, reward):
        self.memory.append([done, action, observation, prev_obs, reward])
        
    def experience_replay(self, update_size=20):
        if (len(self.memory) < update_size):
            return
        else: 
            batch_indices = np.random.choice(len(self.memory), update_size)
            for index in batch_indices:
                done, action_selected, new_obs, prev_obs, reward = self.memory[index]
                action_values = self.forward(prev_obs, remember_for_backprop=True)
                next_action_values = self.forward(new_obs, remember_for_backprop=False)
                experimental_values = np.copy(action_values)

                if (done):
                    experimental_values[action_selected] += self.gamma*reward

                self.backward(action_values, experimental_values)
        self.epsilon = self.epsilon if self.epsilon < self.epsilon_min else self.epsilon*self.epsilon_decay
        for layer in self.layers:
            layer.update_time()
            layer.update_stored_weights()
        
    def backward(self, calculated_values, experimental_values): 
        # values are batched = batch_size x output_size
        delta = (calculated_values - experimental_values)
        # print('delta = {}'.format(delta))
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def train(self, num_episodes, max_steps, exp_upd):
        reward = 0.0
        ep_reward = 0.0
        max_ep_reward = 0.0
        done = False
        exp_upd_i = 0

        last_100_ep_rewards = []
        observation1 = []

        for episode in range(num_episodes):
            observation1 = self.env.reset()
            observation = self.env.reset()

            ep_reward = 0.0

            for step in range(max_steps):

                #print(step, sep=' ', end='-', flush=True)

                action = self.select_action(observation)
                observation1 = observation

                observation, reward, done, info = self.env.step(action)
                #print(action, sep=' ', end='-', flush=True)
                #print(reward, sep=' ', end='-', flush=True)
                #print(step,action,round(reward,2))

                self.remember(done, action, observation, observation1, reward)

                ep_reward += reward

                self.experience_replay(exp_upd)
                #exp_upd_i += 1
                #if (exp_upd_i == exp_upd):
                #    self.experience_replay(exp_upd)
                #    exp_upd_i = 0

                if (done or (step == max_steps-1)):
                    #print()
                    last_100_ep_rewards.append(ep_reward)
                    if (len(last_100_ep_rewards) > 100):
                        last_100_ep_rewards.pop(0)
                    max_ep_reward = np.average(last_100_ep_rewards)
                    print("Episode", episode + 1, "(", 100*episode/num_episodes, "%) has ended after", step + 1, "with reward", round(ep_reward,5), "and net worth", max_ep_reward, " ", self.epsilon)
                    break




