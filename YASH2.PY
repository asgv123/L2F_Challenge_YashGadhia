import gym
import gym_jsbsim
import math
import numpy as np
import matplotlib.pyplot as plt


def random_agent():

    env = gym.make("GymJsbsim-HeadingControlTask-v0")
    state = env.reset()
    x = 0
    y = 0
    step = 1
    plt.axis([-100, 100, -100, 100])
    plt.plot(np.array([0, 100]),np.array([0, 0]), color='green')
    #plt.plot(np.array([0, 100*math.cos((state[0]/180)*math.pi)]),np.array([0, 100*math.sin((state[0]/180)*math.pi)]), color='red')
    plt.scatter(x, y, s=5, c='black')
    plt.pause(0.1)
    done = False

    while not done:

        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        x = x + step*math.cos((state[0]/180)*math.pi)
        y = y + step*math.sin((state[0]/180)*math.pi)
        #plt.plot(np.array([0, 100]),np.array([0, 0]), color='green')
        #plt.plot(np.array([0, 100*math.cos((state[0]/180)*math.pi)]),np.array([0, 100*math.sin((state[0]/180)*math.pi)]), color='red')
        plt.scatter(x, y, s=5, c='black')
        #plt.plot(np.array([x, x+1000*math.cos((state[0]/180)*math.pi)]),np.array([y, y+1000*math.sin((state[0]/180)*math.pi)]), color='red')
        plt.pause(0.1)

        print("action =", action, " ---> State =", state, " : Reward =", reward)
    plt.show()


if __name__ == "__main__":
    random_agent()
