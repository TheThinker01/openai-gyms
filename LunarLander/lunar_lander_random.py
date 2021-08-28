#
#  Just randomly samples actions from the actions space and performs them, comment out `env.render()` to prevent the rendering
#
import gym

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 10

    for i in range(n_games):
        env.reset()
        done = False
        score = 0

        while not done:
            action = env.action_space.sample()
            obs_, reward, done, info = env.step(action)
            env.render()

        print('episode ', i, ' score %.2f' % score)

