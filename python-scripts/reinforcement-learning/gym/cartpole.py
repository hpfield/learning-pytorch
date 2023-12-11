import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple NN
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc = nn.Linear(4, 2) # Input size for cartpole is 4, output is 2 (left or right)

    def forward(self, x):
        # dim=1 specifies applying aoftmax along the second dimension of the output of self.fc
        # in the case of the output size being 2, i.e. [4][1] - dim 0 is 1 (height), dim 1 is 2 (width)
        return torch.softmax(self.fc(x), dim=1) 
    
# Create instance of cartpole env
env = gym.make('CartPole-v1')

# Create instance of local policy network and define optimizer
policy = Policy()
# Adam creates separate learning rates for each param, 
# adapting based on the mean and variance of past gradients
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# Training loop
for episode in range(100):
    state = env.reset()[0]
    done = False
    t= 0 #timesteps
    while not done:
        env.render()
        # The state is read from the environment and is converted into a pytorch tensor
        # The action is determined using only state observations
        state = state.reshape(1, -1)
        action_probs = policy(torch.tensor(state, dtype=torch.float32))
        # multinomial distributions model the likelihood of several different things happening
        # the 1 refers to picking one action from the distribution (highest probability)
        action = torch.multinomial(action_probs, 1).item()
        # The reward function is in the environment in this case
        next_state, reward, done, _, _ = env.step(action)

        # Train
        optimizer.zero_grad() # Resets all gradients of all model params to zero
        # Calculate loss (negative reward for CartPole as it's a maximization task)
        # Taking the negative log probability of the chosen action encourages the model to increase
        # the prob of actions that receive higher rewards. Negative log prob of highly likely actions
        # will be small. The loss being scaled by the reward tells the model that the probability
        # of the chosen action should have been higher/lower
        loss = -torch.log(action_probs[0, action]) * reward
        loss.backward() # compute the gradients
        optimizer.step() # update params based on gradients

        state = next_state
        t += 1

    print(f'Episode {episode + 1} finished after {t+1} timesteps')

env.close()