from helpers import *
from uofgsocsai import LochLomondEnv
import numpy as np
import pandas as pd
import sys
import itertools  
import operator  
from utils import print_table

# We define the random agent and apply it to all task environment ID's
def random_agent(problem_id):

    # should be less than or equal to 0.0, we select 0 because reaching goal state is hard enough for random agent
    reward_hole = 0.0

    # we generate 10 000 episodes in order to give agent chance to reach the goal multiple times
    max_episodes = 10000   

    # every episode should have 2000 iterations (agent can take 2000 steps in the map)
    max_iter_per_episode = 2000 

    # setup the frozen lake loch lomond environment (uncertainty involved) 
    env = LochLomondEnv(problem_id=problem_id, is_stochastic=True, reward_hole=reward_hole)
    results = []

    print('Running Random Agent for problem: ', problem_id)

    for e in range(max_episodes): # iterate over total number of possible episodes
        
        # Reset the random generator to known state (probability needs to adapt)
        np.random.seed(e)

        observation = env.reset() # reset the state of the environment to starting state S     
        
        for iter in range(max_iter_per_episode):
            
            # current agent takes random actions
            action = env.action_space.sample() 

            # outcome of taking a certain action 
            observation, reward, done, info = env.step(action)
          
            # Test condition to see if agent is done and associated rewards 
            if (done and reward==reward_hole): 
                break

            if (done and reward == +1.0):
                break

        results.append([e, iter+1, int(reward)])

    columns = ['episode', 'iterations', 'reward']

    # Save results to a CSV file
    np.savetxt('out_random_{}.csv'.format(problem_id), np.array(results), 
               header="episode,iterations,reward", delimiter=",", fmt='%s')
    
    dataframe = pd.DataFrame(data=np.array(results), index=np.array(results)[0:,0], columns=columns)
    dataframe['cumulative_rewards'] = list(itertools.accumulate(dataframe['reward'], operator.add))
    dataframe['mean_rewards'] = dataframe.apply(lambda x: mean_rewards(x), axis=1)

    # Plotting the results for all task environments ID 0 to 7
    x = range(1, len(dataframe) + 1)
    y = dataframe['mean_rewards']

    title = 'Mean Reward vs Episodes'
    subtitle = 'Random Agent: Problem ID {}'.format(problem_id)
    labels = ['Episodes', 'Mean Reward']
    
    dataframe = dataframe[['episode','iterations','cumulative_rewards','mean_rewards']]

    add_plot(x, y, 'out_random_{}_mean_reward.png'.format(problem_id), title, subtitle, labels)

    print('Total episodes run: ', max_episodes)
    print('Allowed iterations per episode: ', max_iter_per_episode)
    print('Max iterations per episode: ', max(dataframe['iterations']))
    print('Mean iterations per episode: ', dataframe['iterations'].mean())
    print('Average success per episode: ', max(dataframe['cumulative_rewards']) / max_episodes)
    print('Episodes won: ', max(dataframe['cumulative_rewards']))
    print("\n")

    return dataframe
 
# We guarantee that the problem ID is within the allowed bound [0, 7]
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: run_rl.py <problem_id>")
        exit()

    if not (0 <= int(sys.argv[1]) <= 7):
        raise ValueError("Problem ID must be 0 <= problem_id <= 7")        

    random_agent(int(sys.argv[1]))