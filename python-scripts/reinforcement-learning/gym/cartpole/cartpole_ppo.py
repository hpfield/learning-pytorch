import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# PPO: Improve the policy while ensuring the new policy doesn't deviate too far from the old one.

# Hyperparams
GAMMA = 0.99 # Importance of rewards
EPS_CLIP = 0.2 # Provides a threshold for how much the policy can change in a single training update
LR = 0.0003
BATCH_SIZE = 32
K_EPOCHS = 4 # Defines how many times algorithm will iterate over all the experiences per iteration of training

# Create NN for policy and value
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(4, 128) # Extracts features from the input state
        self.policy_layer = nn.Linear(128, 2) # 2 actions to choose from (Actor)
        self.value_layer = nn.Linear(128, 1) # Estimates value of state, predicting expected return from state under current policy (Critic)

    def forward(self, x):
        # When computing gradients, the self.fc layer is influenced by both the actor and the critic
        # This is not considered to breach the separation between the two functions
        # Using a shared layer is a design choice and is not strictly necessary
        x = torch.relu(self.fc(x))
        policy = torch.softmax(self.policy_layer(x), dim=1) # dim=1 is a necessary evil where we must specify the axis which softmax must be applied to
        # The value layer assigns a single value to the state. This value represents the expected return (sum of rewards) form the current state
        value = self.value_layer(x) 
        return policy, value
    
    def evaluate(self, states, actions):
        policy, state_values = self.forward(states)
        # Place probability distribution into a Categorical object for RL fucnitons
        dist = torch.districutions.Categorical(policy)
        # Compute log probs of actions
        # Computing the log prob of each action allows for the gradient to be scaled by the 
        # probability of an action, encouraging the policy to increase the likelihood of 
        # beneficial actions. Log probs are more numerically stable than using just the probs.
        # Log probs are also used to calculate the entropy of a policy, which is a measure of 
        # randomness.
        log_probs = dist.log_prob(actions)

        return log_probs, state_values


#! Monte carlo method, reward = action-value function
def compute_returns(rewards, masks, gamma):
    returns = []
    R = 0
    # Iterates through each timestep in reverse, starting from the end of the episode
    # For each timestep, the reward is the current reward plus the discounted reward from the next timestep
    # The mask enables this to span multiple episodesin a single batch, as the R value is reset when an 
    # episode terminates
    for reward, mask in zip(reversed(rewards), reversed(masks)):
        R = reward + gamma * R * mask
        # insert() function un-reverses the order, ensuring that each new R value is stored at the 
        # beginning of the list.
        returns.insert(0, R) 
    return returns

def ppo_update(policy, optimizer, memory, ppo_epochs, batch_size, clip_param):
    for _ in range(ppo_epochs):
        for states, actions, old_log_probs, returns, advantage in memory.sample(batch_size):
            # Evaluate new policy
            new_log_probs, state_values = policy.evaluate(states, actions)

            # Calculate the advantage
            # The difference between the actual return and the expected return (state value)
            # Positive means better than expected and vice versa
            # returns: total accumulated rewards from a given timestep until the end of an episode
            # state_values: values assigned to individual states with policy.fc and policy.value_layer
            #! Use of detach()
            # By default, torch tracks all computations in a computation graph, and detach stops state_values being recorded in the advantage tensor
            # If detach() was omitted, everything used to create state_values would be included in the advantage
            # This becomes relevant when loss.backward() is called later
            # If detach is not used, the gradients are propogated through the actor and critics networks
            # with respect to each other, which violates their separation. 
            # The resulting issue is to update the actor's parameters, treating the critics output as a variable
            # that influences the policy's loss
            #! Monte-carlo method
            advantage = returns - state_values.detach()

            # Calculate the ratios
            # Calculating the exponential of the new_log_probs - old_log_probs is evuivalent 
            # to new_log_probs/old_log_probs
            # The ratios quatify how much the actions of the policy have changed since the last update
            ratios = torch.exp(new_log_probs - old_log_probs)

            # Calculate Surrogate losses
            # Scale the ratios by the advantage
            # Which means to scale the change in the policy by how good the policy decisions were
            # found to be compared to the average (which is just the state value)
            surr1 = ratios * advantage
            # torch.clamp(input, min, max) constrains values of a tensor within a specific range
            # By making sure that the ratio is not too far from 1 (where they would be the same)
            # the amount by which the policy changes can be controlled. 
            # This is also scaled by the advantage
            surr2 = torch.clamp(ratios, 1-clip_param, 1+clip_param) * advantage

            # Policy gradient loss
            # The minimum value for each element in surr1 and surr2 is chosen and placed into a new tensor
            # When all the minimum values have been chosen, the mean of the whole tensor is taken as the loss
            # The minimum is chosen so that the most conservative update is selected
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            # Calculating the MSE between the state_values and the returns
            # We want to optimise the state value function so that it more closely approximates the reward
            value_loss = nn.MSELoss()(state_values, returns)

            # Total loss
            # Combining policy and value loss is common in actor-critic methods with two objectives
            # adding the 0.5 coefficient is supposed to balance the two components
            # The size of the coefficient places a corresponding emphasis on accurately predicting that component
            # In this case the policy is emphasised, but this can be adjusted to suit the environment
            loss = policy_loss + 0.5 * value_loss

            # Take gradient step
            optimizer.zero_grad()
            loss.backward() # Computes gradients for both the actor and critic
            optimizer.step() # Applies gradients and takes learning step



    
def main():
    # Create instance of cartpole env
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    policy = ActorCritic()
    optimizer = optim.Adam(policy.parameters(), lr=LR)
 
    memory = Memory() # need to define memory class to store experiences

    # Training loop
    for episode in range(10000):
        state = env.reset()[0]
        terminated = False
        truncated = False
        t= 0 #timesteps
        while not terminated or truncated:
            state = torch.tensor(state, dtype=torch.float32) # Convert state to torch tensor
            action_probs, _ = policy(state) # Get vector of probs over actions using policy
            dist = Categorical(action_probs) # Doesn't change vector, allows use of sample()
            action = dist.sample() # Randomly choose an action according to the probability dist

            next_state, reward, terminated, truncated, _ = env.step(action.item())

            # Store in memory
            memory.states.append(state)
            memory.actions.append(action)
            memory.log_probs.append(dist.log_prob(action))
            memory.rewards.append(reward)
            memory.is_terminals.append(terminated or truncated)

            state = next_state
            t +=1

            # Update if batch is full
            if len(memory) == BATCH_SIZE:
                # Batch size defines how much data will be used for return computation
                returns = compute_returns(memory.rewards, memory.is_terminals, GAMMA)
                memory.returns = returns
                memory.compute_advantages() # need to implement
                ppo_update(policy, optimizer, memory, K_EPOCHS, BATCH_SIZE, EPS_CLIP)
                memory.clear()

        print(f'Episode {episode + 1} ended after {t+1} timesteps')

    # Save the model
    torch.save(policy.state_dict(), 'ppo_cartpole.pth')

if __name__ == '__main__':
    main()