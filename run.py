import gym
import numpy as np
import statistics
import q_learning_agent
import envs.environment

# functions

q_agent = q_learning_agent.QLearningAgent()

# end of functions

env = gym.make('MDPAlireza-v0')

# Number of episodes for the agent to run

# parameters needed by our policy and learning rule
params = {
  'epsilon': 0.1,  # epsilon-greedy policy
  'alpha': 0.1,  # learning rate
  'gamma': 0.5,  # discount factor
}

# episodes/trials
n_episodes = 100
max_steps = 200000

results = q_agent.learn_environment(env,q_agent.q_learning, params, max_steps, n_episodes)
value_qlearning, reward_sums_qlearning, steps = results

print(steps)
print(value_qlearning)



