import gym
import numpy as np
import matplotlib.pyplot as plt
from reinforce_agent import PolicyGradientAgent

# plots the running average of 100 scores obtained by the agent vs the episode number 
def plot_learning_curves(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x,running_avg)
    plt.title('Running Average of the last 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    # Setup parameters
    n_episodes = 3000
    gamma = 0.99
    lr = 0.005
    input_dims = env.observation_space.shape
    n_actions = env.action_space.n
    
    # create the Agent
    agent = PolicyGradientAgent(gamma=gamma, input_dims=input_dims , n_actions= n_actions, lr = lr)
    print('The detected device is ', agent.policy.device)
    
    # the file name for storing the final plot
    fname = 'REINFORCE_' + 'lunar_lander_lr' + str(agent.lr) + '_' \
            + str(n_episodes) + 'episodes'
    figure_file = 'plots/' + fname + '.png'

    scores = []
    best_avg = -1000
    # loop over all the episodes 
    for i in range(n_episodes):
        done = False
        observation = env.reset()
        score = 0

        # Run one episode 
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score+=reward
            observation = observation_
            agent.store_rewards(reward)
            # uncomment to render the episode 
            # env.render()

        # Episode completed , now call learn to update the policy
        agent.learn()
        
        # terminal data for debugging and progress monitoring
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score)

        # save the model when we arrrive at a new best score 
        # if avg_score>=best_avg:
        #     best_avg = avg_score
        #     T.save(agent.)
    # All done now plot the figure 
    x = [i+1 for i in range(len(scores))]
    plot_learning_curves(scores, x)



