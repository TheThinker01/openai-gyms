import numpy as np

class ReplayBuffer():
    # n_actions is bascially the number of components to the action eg :  
    #           For Lunar Lander thrust is a vector of four components , left, right , top, bottom
    # input_shape is the shape of the observations from the environment
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        
        self.mem_cntr = 0       # total number of entires added to the replay buffer 
                                # Note that mem_cntr  is the total number of entries and not the index
        
        # We store the <state, action, reward, state_, done> pentuple
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        # find the index which will be replace with new data 
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    # Sample the agents buffer uniformly
    def sample_buffer(self, batch_size):
        # maximum range to draw from, we dont want to draw the initialized zeros 
        max_mem = min(self.mem_cntr, self.mem_size)

        # generates a random sample of size batch_size 
        # from a range of size max_mem 
        batch = np.random.choice(max_mem, batch_size)

        # get the data to return
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones





