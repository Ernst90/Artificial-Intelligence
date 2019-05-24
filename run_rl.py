from helpers import *
from uofgsocsai import LochLomondEnv
import numpy as np
import pandas as pd
import sys
import itertools  
import operator  
from search import *
from utils import print_table

class QLearningAgentUofG:
    """ Here a model free approach for our reinforcement learning agent will be defined. Q-learning is used for this, 
        which maximizes the EV of the total reward over all iterations and finds a policy based on it. Q-learning 
        involves a stochastic environment, but since the agent "learns" over time through exploration, with increasing 
        number of episodes actions are taken more informed (deterministic). It uses Q-value iteration in an temporal 
        difference framework. 
    """

    def __init__(self, terminals, all_act, alpha, gamma, Ne, Rplus):

        self.gamma = gamma
        self.alpha = alpha
        self.terminals = terminals
        self.all_act = all_act
        self.Q = defaultdict(float)
        self.Nsa = defaultdict(float)        
        self.s = None
        self.a = None
        self.r = None
        self.Ne = Ne
        self.Rplus = Rplus

    def f(self, u, n):       
        """ Exploration function to obtain an improved estimate of the optimal Q-function. That is, 
            actions are considered which may differ from the action that is thought to be currently best. 
        """

        # if n < self.Ne:
        #    return self.Rplus

        return u

    def actions_in_state(self, state):
        """ Return actions possible in given state.
            Useful for max and argmax. """
        if state in self.terminals:
            return [None]
        else:
            return self.all_act

    def __call__(self, new_state, new_reward, episode):
        alpha, gamma, terminals = self.alpha, self.gamma, self.terminals
        Q, Nsa = self.Q, self.Nsa
        s, a, r = self.s, self.a, self.r

        if a is not None:
            Nsa[s, a] += 1
            Q[s, a] += alpha(Nsa[s, a]) * (r + gamma * max(Q[new_state, a1] for a1 in self.actions_in_state(new_state)) - Q[s, a])

        if new_state in terminals:
            self.Q[new_state, None] = new_reward
            self.s = self.a = self.r = None
        else:
            self.s, self.r = new_state, new_reward    
            self.a = argmax(self.actions_in_state(new_state), key=lambda a1: self.f(Q[new_state, a1], Nsa[s, a1]))

            if random.uniform(0, 1) <  1/(episode+1):
                self.a = random.randint(0, len(self.all_act)-1)            

        return self.a

    def update_state(self, percept):
        """To be overridden in most cases. The default case
        assumes the percept to be of type (state, reward)."""
        return percept

# Define the reinforcement agent and apply it to all task environment ID's
def rl_agent(problem_id):

    # select small negative rewards for the RL-Agent to create an incenctive to learn
    reward_hole = -0.01

    # generate 10 000 episodes in order to give agent chance to reach the goal
    max_episodes = 10000   

    # every episode should have 2000 iterations (agent can take 2000 steps in the map)
    max_iter_per_episode = 2000 

    results = []

    # setup the frozen lake loch lomond environment (uncertainty involved)
    env = LochLomondEnv(problem_id=problem_id, is_stochastic=True, reward_hole=reward_hole)
    all_act = list(range(env.action_space.n))

    q_agent = QLearningAgentUofG(terminals=get_terminals(env), all_act=all_act, alpha=lambda n: 0.8, gamma=0.8, Rplus=2, Ne=5)

    print('Running Q Learning Agent for problem: '.format(problem_id))
    print("(will take a while)")
    
    for e in range(max_episodes): # iterate over episodes

        state = env.reset() # reset the state of the environment to starting state S     
        reward = 0
        
        # Over the total number of allowed iterations  
        for iter in range(max_iter_per_episode):
            action = q_agent(state, reward, e+1)
            
            # current agent takes actions
            if action is not None:
                state, reward, done, info = env.step(action) 

            # Test condition to see if agent is done and associated rewards
            if done:
                q_agent(state, reward, e+1)

                break

        results.append([e, iter+1, int(reward)])

    # Compute the policy
    policy = {}

    for state_action, value in list(q_agent.Q.items()):
        state, action = state_action
        policy[state] = argmax(q_agent.actions_in_state(state), key=lambda a: q_agent.Q[state, a])

    print('Policy: ')
    print_table(to_arrows(policy, 8, 8))

    # Save results to a CSV file
    np.savetxt('out_rl_{}.csv'.format(problem_id), np.array(results), 
               header="episode,iterations,reward", delimiter=",", fmt='%s')

    np.savetxt('out_rl_{}_policy.txt'.format(problem_id), to_arrows(policy, 8, 8), delimiter="\t", fmt='%s')

    # Add a plot over all 10 000 episodes 
    columns = ['episode', 'iterations', 'reward']

    dataframe = pd.DataFrame(data=np.array(results), index=np.array(results)[0:,0], columns=columns)
    dataframe['cumulative_rewards'] = list(itertools.accumulate(dataframe['reward'], operator.add))
    dataframe['mean_rewards'] = dataframe.apply(lambda x: mean_rewards(x), axis=1)

    x = range(1, len(dataframe) + 1)
    y = dataframe['mean_rewards']
    
    title = 'Mean Reward vs Episodes'
    subtitle = 'RL-Agent: Problem ID {}'.format(problem_id)
    labels = ['Episodes', 'Mean Reward']
    
    add_plot(x, y, 'out_rl_{}.png'.format(problem_id), title, subtitle, labels)

    # Adding plot for the last 1000 episodes to detect potential learning 
    dataframe_ac = pd.DataFrame(data=np.array(results)[range(max_episodes-1000, max_episodes),:], columns=columns)
    dataframe_ac['episode'] = range(1000)
    dataframe_ac['cumulative_rewards'] = list(itertools.accumulate(dataframe_ac['reward'], operator.add))
    dataframe_ac['mean_rewards'] = dataframe_ac.apply(lambda x: mean_rewards(x), axis=1)

    x = range(1, len(dataframe_ac) + 1)
    y = dataframe_ac['mean_rewards']

    title = 'RL-Agent: Problem ID {}'.format(problem_id)
    subtitle = 'Last 1000 Episodes'
    labels = ['Last 1000 Episodes', 'Mean Reward']
    
    add_plot(x, y, 'out_rl_{}_converged.png'.format(problem_id), title, subtitle, labels)

    # Print involved performance measures over all 10 000 episodes 
    print('Total episodes run: ', max_episodes)
    print('Allowed iterations per episode: ', max_iter_per_episode)
    print('Max iterations per episode: ', max(dataframe['iterations']))
    print('Mean iterations per episode: ', dataframe['iterations'].mean())
    print('Average success per episode: ', max(dataframe['cumulative_rewards']) / max_episodes)
    print('Episodes won: ', max(dataframe['cumulative_rewards']))
    
    # Print involved performance measures over the last 1000 episodes  
    print("\n\n")
    print('Stats for the last 1000 episodes....')
    print('Max iterations per episode: ', max(dataframe_ac['iterations']))
    print('Mean iterations per episode: ', dataframe_ac['iterations'].mean())
    print('Average success per episode: ', max(dataframe_ac['cumulative_rewards']) / 1000)
    print('Episodes won: ', max(dataframe_ac['cumulative_rewards']))

    return dataframe

# Guarantee that the problem ID is within the allowed bound [0, 7]
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: run_rl.py <problem_id>")
        exit()
    
    if not (0 <= int(sys.argv[1]) <= 7):
        raise ValueError("Problem ID must be 0 <= problem_id <= 7")        
 
    rl_agent(int(sys.argv[1]))
