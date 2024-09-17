import gym
import numpy as np
import statistics
import q_learning_agent
import envs.environment

# functions

# parameters needed by our policy and learning rule
params = {
  'epsilon': 0.1,  # epsilon-greedy policy
  'alpha': 0.1,  # learning rate
  'gamma': 0.5,  # discount factor
  'k': 2,  # number of planning steps
  'trans_prior': 0.01,  # prior for transition probabilities
  'pure_novelty': False,  # whether to use novelty reward
  'pure_surprise': True,  # whether to use surprise reward
}

# end of functions

env = gym.make('MDPAlireza-v0')
q_agent = q_learning_agent.QLearningAgent(env, params)


# Number of episodes for the agent to run



# episodes/trials
n_episodes = 1
max_steps = 10000

results = q_agent.learn_environment(env, q_agent.dyna_q_model_update, q_agent.dyna_q_planning, q_agent.q_learning, params, max_steps, n_episodes)
value_qlearning, reward_sums_qlearning, steps = results

print(steps)
print(value_qlearning)



