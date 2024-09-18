import gym
import numpy as np
import statistics
import q_learning_agent
import envs.environment
import matplotlib.pyplot as plt

# functions
agent_dict = {"surprise": [],
              "novelty": []}

# parameters needed by our policy and learning rule
params_surprise = {
  'epsilon': 0.1,  # epsilon-greedy policy
  'alpha': 0.1,  # learning rate
  'gamma': 0.5,  # discount factor
  'k': 2,  # number of planning steps
  'trans_prior': 0.01,  # prior for transition probabilities
  'pure_novelty': False,  # whether to use novelty reward
  'pure_surprise': True,  # whether to use surprise reward
}

params_novelty = {
    'epsilon': 0.1,    # epsilon-greedy policy
    'alpha': 0.1,      # learning rate
    'gamma': 0.5,      # discount factor
    'k': 5,            # number of planning steps
    'trans_prior': 0.01,   # prior for transition probabilities
    'pure_novelty': True,  # whether to use novelty reward
    'pure_surprise': False # whether to use surprise reward
}

# episodes/trials
n_episodes = 1
max_steps = 10000

# end of functions

env = gym.make('MDPAlireza-v0')

# surprise agent
q_agent_surprise = q_learning_agent.QLearningAgent(env, params_surprise)
results_surprise = q_agent_surprise.learn_environment(env, q_agent_surprise.dyna_q_r_f_update, q_agent_surprise.dyna_q_planning, q_agent_surprise.q_learning, params_surprise, max_steps, n_episodes)
surprise_counts = q_agent_surprise.novelty_count
agent_dict["surprise"] = surprise_counts.copy()

# novelty agent
q_agent_novelty = q_learning_agent.QLearningAgent(env, params_novelty)
results_novelty = q_agent_novelty.learn_environment(env, q_agent_novelty.dyna_q_r_f_update, q_agent_novelty.dyna_q_planning, q_agent_novelty.q_learning, params_novelty, max_steps, n_episodes)
novelty_counts = q_agent_novelty.novelty_count
agent_dict["novelty"] = novelty_counts.copy()


states = np.arange(len(novelty_counts))  # Assuming the state space is 1D

# Plot bar graphs for novelty and surprise visit counts
width = 0.35  # width of the bars
fig, ax = plt.subplots()

novelty_bars = ax.bar(states - width/2, agent_dict["novelty"], width, label='Novelty Agent')
surprise_bars = ax.bar(states + width/2, agent_dict["surprise"], width, label='Surprise Agent')

# Add some text for labels, title, and axes ticks
ax.set_xlabel('State')
ax.set_ylabel('Visit Counts')
ax.set_title('State visit counts for Novelty vs Surprise Agent')
ax.set_xticks(states)
ax.legend()

# Show plot
plt.show()

