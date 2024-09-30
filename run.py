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
n_runs = 2  # Number of times to run each agent
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
    'random': False
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
    'stoch': stoch,
    'random': False
}

params_random = {
    'epsilon': 0.1,  # epsilon-greedy policy
    'alpha': 0.1,  # learning rate
    'gamma': 0.25,  # discount factor
    'k': 2,  # number of planning steps
    'trans_prior': 0.01,  # prior for transition probabilities
    'pure_novelty': False,  # whether to use novelty reward
    'pure_surprise': False,  # whether to use surprise reward
    'temperature': 0.3,  # temperature for softmax policy
    'stoch': stoch,
    'random': True
}



# List of trial segments
segments = ['10', '20', '30', '40', '50', '100', '200', '500', '1000', '2000', '5000', '10000']  # Trial segments
if not stoch:
    states = np.arange(11)  # Assuming there are 11 states, adjust this as needed
else:
    states = np.arange(61)

overall_surprise_counts = {segment: np.zeros(len(states)) for segment in segments}
overall_novelty_counts = {segment: np.zeros(len(states)) for segment in segments}
overall_random_counts = {segment: np.zeros(len(states)) for segment in segments}

segment_surprise_counts = {segment: np.zeros(len(states)) for segment in segments}
segment_novelty_counts = {segment: np.zeros(len(states)) for segment in segments}
segment_random_counts = {segment: np.zeros(len(states)) for segment in segments}


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

    # Run random agent
    q_agent_random = q_learning_agent.QLearningAgent(env, params_random)
    q_agent_random.learn_environment(env, q_agent_random.dyna_q_r_f_update, q_agent_random.prioritized_planning,
                                        q_agent_random.q_learning, params_random, max_steps, n_episodes)
    random_counts = q_agent_random.step_count_dict

    for segment in segments:
        overall_surprise_counts[segment] = np.add(overall_surprise_counts[segment], [surprise_counts[segment]],
                                                  casting='unsafe')
        overall_novelty_counts[segment] = np.add(overall_novelty_counts[segment], [novelty_counts[segment]],
                                                 casting='unsafe')
        overall_random_counts[segment] = np.add(overall_random_counts[segment], [random_counts[segment]],
                                                casting='unsafe')


    # Accumulate the state visit counts for each segment
for i, segment in enumerate(segments):
    overall_novelty_counts[segment] = np.divide(overall_novelty_counts[segment], n_runs)
    overall_surprise_counts[segment] = np.divide(overall_surprise_counts[segment], n_runs)
    overall_random_counts[segment] = np.divide(overall_random_counts[segment], n_runs)

    if i == 0:
        # For the first segment, just divide by the number of runs as no previous segments exist
        segment_novelty_counts[segment] = overall_novelty_counts[segment]
        segment_surprise_counts[segment] = overall_surprise_counts[segment]
        segment_random_counts[segment] = overall_random_counts[segment]
    else:
        # For subsequent segments, subtract the cumulative average of previous segments
        previous_segment = segments[i - 1]

        segment_novelty_counts[segment] = overall_novelty_counts[segment] - overall_novelty_counts[previous_segment]
        segment_surprise_counts[segment] = overall_surprise_counts[segment] - overall_surprise_counts[previous_segment]
        segment_random_counts[segment] = overall_random_counts[segment] - overall_random_counts[previous_segment]

plot_stacked_individual_segments(segment_surprise_counts, segment_novelty_counts, segment_random_counts, segments, stoch, output_dir)








