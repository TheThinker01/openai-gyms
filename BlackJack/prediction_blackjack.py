import numpy as np
# class for the agent
class Agent():
    def __init__(self, gamma=0.99):
        self.V = {}                                            # Value for all the states
        self.sum_space = [i for i in range(4,22)]              # sums that the user can obtain
        self.dealer_show_card_space = [i+1 for i in range(10)] # card values that can be shown by the user 
        self.ace_space = [False, True]                         # whether the user has a usable ace or not
        self.action_space = [0,1]                              # Stick or Hit

        self.state_space = []                                  # 3 tuple : (sum, dealer_card, usable_ace)
        self.returns = {}                                      # returns that follow first visit to each state
        self.states_visited = {}                               # first vist or not
        self.memory = []                                       # states visited and returns received
        self.gamma = gamma

        self.init_vals()

    def init_vals(self):
        for total in self.sum_space:
            for card in self.dealer_show_card_space:
                for ace in self.ace_space:
                    self.V[(total, card, ace)] = 0
                    self.returns[(total, card, ace)] = []
                    self.states_visited[(total, card, ace)] = 0
                    self.state_space.append((total, card, ace))

    def policy(self, state):
        total, _, _ = state
        action = 0 if total >= 20 else 1
        return action

    def update_V(self):
        # calculate the return for each state using rewards from memory
        for idt, (state, _) in enumerate(self.memory):
            G = 0
            if self.states_visited[state] == 0:
                self.states_visited[state] += 1
                discount = 1
                for t, (_, reward) in enumerate(self.memory[idt:]):
                    G += reward*discount
                    discount *= self.gamma
                    self.returns[state].append(G)

        # update the Value function for state using the mean of the rewards 
        for state, _ in self.memory:
            self.V[state] = np.mean(self.returns[state])

        # reset the visited list for the next episode 
        for state in self.state_space:
            self.states_visited[state] = 0

        # reset the memory for the next episode 
        self.memory = []
