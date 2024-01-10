import torch

class Memory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.returns = []
    
    def __len__(self):
        return len(self.states)
    
    # Generate random mini-batch samples from the memory to train the network
    def sample(self, batch_size):
        # i starts at 0, goes up to self.states, and steps by size batch_step each time
        # each iteration processes one mini-batch of data
        for i in range(0, len(self.states), batch_size):
            # The yield python generator mechanism allows values to be computed over time
            # rather than all at once and storing in memory
            # Returns a tuple of tensors representing a mini-batch of states, actions, log_probs and returns
            # Turns the sample function into a generator, where a tuple is only generated when requested
            # (More memory efficient, especially with large datasets)
            yield (
                # Slicing: selects subset of the data from index i, up to but not including index i+batch_size
                # Creating tensors: torch.stack() takes a list of tensors and combines them into a new tensor
                # In this case, the slices are combined into a tensor
                # Each slice is an entry along the 1st dim of the combined tensor
                # Slices are still kept separate, but are now in a manipulable tensor format
                torch.stack(self.states[i:i+batch_size]),
                torch.stack(self.actions[i:i+batch_size]),
                torch.stack(self.log_probs[i:i+batch_size]),
                torch.stack(self.returns[i:i+batch_size])
            )
            