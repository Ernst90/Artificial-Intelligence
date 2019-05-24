from helpers import *
from uofgsocsai import LochLomondEnv
import numpy as np
import pandas as pd
import sys
import itertools  
import operator  
from search import *

# Define the simple agent (A*) and apply it to all task environment ID's
def simple_agent(problem_id):

    # since A*star agent is fully informed, any negative reward for hole would not make a difference, hence we chose 0
    reward_hole = 0.0

    # generate 10 000 episodes in order to give agent chance to reach the goal multiple times
    max_episodes = 10000   

    # since A*star agent always wins, lower limit for allowed iterations per episode to 100 (time constraint)
    max_iter_per_episode = 100 
 
    actions = []
    results = []
 
    # setup the frozen lake loch lomond environment (deterministic, no uncertainty)
    env = LochLomondEnv(problem_id=problem_id, is_stochastic=False, reward_hole=reward_hole)
    state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)
    
    # informed search problem 
    undirected_graph = UndirectedGraph(state_space_actions)
    undirected_graph.locations = state_space_locations
    graph_problem = GraphProblem(state_initial_id, state_goal_id, undirected_graph)

    node = astar_search(problem=graph_problem, h=None)
    best_path = node.solution()

    print('Running Simple Agent for problem: '.format(problem_id))

    for i in range(len(best_path)):
        if i == 0:
            previous = undirected_graph.locations[state_initial_id]
        else:
            previous = undirected_graph.locations[best_path[i - 1]]

        current = undirected_graph.locations[best_path[i]]

        action = get_action_from_location(previous, current)
        actions.append(action)

    for e in range(max_episodes): # iterate over total number of possible episodes

        observation = env.reset() # reset the state of the environment to starting state S
        
        for iter in range(max_iter_per_episode):
            
            # select action from the solution
            action = actions[iter]

            # outcome of taking a certain action
            observation, reward, done, info = env.step(action)
          
            # Test condition to see if agent is done and associated rewards
            if (done and reward==reward_hole): 
                break

            if (done and reward == +1.0):
                break

        results.append([e, iter+1, int(reward)])

    # Save results to a CSV file
    np.savetxt('out_simple_{}.csv'.format(problem_id), np.array(results), 
               header="episode,iterations,reward", delimiter=",", fmt='%s')

    columns = ['episode', 'iterations', 'reward']
    
    dataframe = pd.DataFrame(data=np.array(results), index=np.array(results)[0:,0], columns=columns)
    dataframe['cumulative_rewards'] = list(itertools.accumulate(dataframe['reward'], operator.add))
    dataframe['mean_rewards'] = dataframe.apply(lambda x: mean_rewards(x), axis=1)
    
    # Plotting the results for all task environments ID 0 to 7
    x = range(1, len(dataframe) + 1)
    y = dataframe['mean_rewards']
    
    title = 'Mean Reward vs Episodes'
    subtitle = 'Simple Agent: Problem ID {}'.format(problem_id)
    labels = ['Episodes', 'Mean Reward']

    add_plot(x, y, 'out_simple_{}_mean_reward.png'.format(problem_id), title, subtitle, labels)
    
    # Print involved performance measures over all 10 000 episodes
    print('Total episodes run: ', max_episodes)
    print('Allowed iterations per episode: ', max_iter_per_episode)
    print('Max iterations per episode: ', max(dataframe['iterations']))
    print('Mean iterations per episode: ', dataframe['iterations'].mean())
    print('Average success per episode: ', max(dataframe['cumulative_rewards']) / max_episodes)
    print('Episodes won: ', max(dataframe['cumulative_rewards']))
    print("\n")

    return dataframe

# Guarantee that the problem ID is within the allowed bound [0, 7]
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: run_simple.py <problem_id>")
        exit()
    
    if not (0 <= int(sys.argv[1]) <= 7):
        raise ValueError("Problem ID must be 0 <= problem_id <= 7")        

    simple_agent(int(sys.argv[1]))
