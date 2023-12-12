import torch
import torch.nn as nn
import numpy as np

# Q stands for 'quality'
class Q:
    # Discount factor (gamma) scales the importance of the rewards in the update equation
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Q table initialisation
        # We take a Q-value for each state-action pair
        # Q value is expected reward if the observed action is observed in the given state, 
        # and then the policy is followed thereafter
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        # Epsilon greedy action selection
        # Epsilon typically set between 0 and 1 (determines degree of exploration vs exploitation)
        # Exploration is random actions, exploitation is using learned experience to choose an action
        if np.random.uniform(0,1) < self.epsilon: # value of espilon will determine how often we explore vs exploit
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[state, :]) # The max value from the Q-table for the given state is the agents best guess
    
    # Update the q-values in the q-table based on the agent's experience transitioning from one state to another
    def update(self, state, action, reward, next_state):
        # Q-learning update rule
        # If no value for next_state, Q-value is 0
        best_next_action = np.argmax(self.q_table[next_state, :]) # Choosing action that minimises q-value for next state
        # Updating the q-value for the current state-action pair based on the difference between:
        # - the estimated optimal future reward
        # - the current estimate
        self.q_table[state, action] += self.learning_rate * (
            reward + self.discount_factor * self.q_table[next_state, best_next_action] - self.q_table[state, action]
        )
    
