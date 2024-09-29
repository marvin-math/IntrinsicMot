import gym
import numpy as np
import statistics
import q_learning_agent
import envs.environment
import matplotlib.pyplot as plt
import seaborn as sns
import os
from plots import plot_stacked_individual_segments

# Directory to save the plots
output_dir = '/Users/marvinmathony/Documents/RLandNNs/plots'
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# episodes/trials
n_episodes = 1
max_steps = 10000
n_runs = 10  # Number of times to run each agent
stoch = True

# parameters needed by our policy and learning rule
params_surprise = {
    'epsilon': 0.1,  # epsilon-greedy policy
    'alpha': 0.1,  # learning rate
    'gamma': 0.25,  # discount factor
    'k': 2,  # number of planning steps
    'trans_prior': 0.01,  # prior for transition probabilities
    'pure_novelty': False,  # whether to use novelty reward
    'pure_surprise': True,  # whether to use surprise reward
    'temperature': 0.3,  # temperature for softmax policy
    'stoch': stoch,
}

params_novelty = {
    'epsilon': 0.1,  # epsilon-greedy policy
    'alpha': 0.1,  # learning rate
    'gamma': 0.25,  # discount factor
    'k': 2,  # number of planning steps
    'trans_prior': 0.01,  # prior for transition probabilities
    'pure_novelty': True,  # whether to use novelty reward
    'pure_surprise': False,  # whether to use surprise reward
    'temperature': 0.3,  # temperature for softmax policy
    'stoch': stoch
}



# List of trial segments
segments = ['10', '20', '30', '40', '50', '100', '200', '500', '1000', '2000', '5000', '10000']  # Trial segments
if not stoch:
    states = np.arange(11)  # Assuming there are 11 states, adjust this as needed
else:
    states = np.arange(61)

overall_surprise_counts = {segment: np.zeros(len(states)) for segment in segments}
overall_novelty_counts = {segment: np.zeros(len(states)) for segment in segments}


# Run both agents 100 times
for run in range(n_runs):
    # Create the environment

    env = gym.make('MDPAlireza-vs') if stoch else gym.make('MDPAlireza-v0')

    # Run surprise agent
    q_agent_surprise = q_learning_agent.QLearningAgent(env, params_surprise)
    q_agent_surprise.learn_environment(env, q_agent_surprise.dyna_q_r_f_update, q_agent_surprise.prioritized_planning,
                                       q_agent_surprise.q_learning, params_surprise, max_steps, n_episodes)
    surprise_counts = q_agent_surprise.step_count_dict


    # Run novelty agent
    q_agent_novelty = q_learning_agent.QLearningAgent(env, params_novelty)
    q_agent_novelty.learn_environment(env, q_agent_novelty.dyna_q_r_f_update, q_agent_novelty.prioritized_planning,
                                      q_agent_novelty.q_learning, params_novelty, max_steps, n_episodes)
    novelty_counts = q_agent_novelty.step_count_dict
    for segment in segments:
        print(f"novelty counts in segment {segment}: {novelty_counts[segment]}")
        overall_surprise_counts[segment] = np.add(overall_surprise_counts[segment], [surprise_counts[segment]],
                                                  casting='unsafe')
        overall_novelty_counts[segment] = np.add(overall_novelty_counts[segment], [novelty_counts[segment]],
                                                 casting='unsafe')

    # Accumulate the state visit counts for each segment
for segment in segments:
    overall_novelty_counts[segment] = np.divide(overall_novelty_counts[segment], n_runs)
    overall_surprise_counts[segment] = np.divide(overall_surprise_counts[segment], n_runs)

plot_stacked_individual_segments(stoch, segments, overall_surprise_counts, overall_novelty_counts, n_runs, output_dir)








