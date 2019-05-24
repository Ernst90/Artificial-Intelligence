from run_random import random_agent
from run_simple import simple_agent
from run_rl import rl_agent
import sys
import matplotlib.pyplot as plt
from helpers import *


# Here we will compare all three agents with each 
def main(problem_id):

    random_dataframe = random_agent(problem_id)
    simple_dataframe = simple_agent(problem_id)
    rl_dataframe = rl_agent(problem_id)

    labels = ['Episodes', 'Mean Reward']
    filename = 'out_{}'.format(problem_id)
    title = 'Mean Reward vs Episodes'
    subtitle = 'Problem ID {}'.format(problem_id)

# Plotting the results for all task environments ID 0 to 7
    plt.plot(random_dataframe['mean_rewards'], '#ff0000', label='Random Agent') # coloured in red
    plt.plot(simple_dataframe['mean_rewards'], '#8e44ad', label='Simple Agent') # coloured in magenta
    plt.plot(rl_dataframe['mean_rewards'], '#15c45d', label='RL-Agent')         # coloured in green
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.suptitle(title)
    plt.title(subtitle)
    plt.legend(loc='best') # automatically chose legend position
    plt.savefig(filename)
    plt.close()

# We guarantee that the problem ID is within the allowed bound [0, 7]
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: run_rl.py <problem_id>")
        exit()
    
    if not (0 <= int(sys.argv[1]) <= 7):
        raise ValueError("Problem ID must be 0 <= problem_id <= 7")        

    main(int(sys.argv[1]))

