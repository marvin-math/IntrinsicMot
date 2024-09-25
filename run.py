import gym
import numpy as np
import statistics
import q_learning_agent
import envs.environment
import matplotlib.pyplot as plt
import seaborn as sns

# parameters needed by our policy and learning rule
params_surprise = {
    'epsilon': 0.1,  # epsilon-greedy policy
    'alpha': 0.1,  # learning rate
    'gamma': 0.5,  # discount factor
    'k': 5,  # number of planning steps
    'trans_prior': 0.01,  # prior for transition probabilities
    'pure_novelty': False,  # whether to use novelty reward
    'pure_surprise': True,  # whether to use surprise reward
}

params_novelty = {
    'epsilon': 0.1,  # epsilon-greedy policy
    'alpha': 0.1,  # learning rate
    'gamma': 0.5,  # discount factor
    'k': 5,  # number of planning steps
    'trans_prior': 0.01,  # prior for transition probabilities
    'pure_novelty': True,  # whether to use novelty reward
    'pure_surprise': False  # whether to use surprise reward
}

# episodes/trials
n_episodes = 1
max_steps = 10000
n_runs = 10  # Number of times to run each agent

# List of trial segments
segments = ['50', '100', '200', '500', '1000', '2000', '5000', '10000']
states = np.arange(11)  # Assuming there are 11 states, adjust this as needed

# Initialize dictionaries to store average counts for each agent
average_surprise_counts = {segment: np.zeros(len(states)) for segment in segments}
average_novelty_counts = {segment: np.zeros(len(states)) for segment in segments}

# Run both agents 100 times
for run in range(n_runs):
    # Create the environment
    env = gym.make('MDPAlireza-v0')

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

    # Accumulate the state visit counts for each segment
    for segment in segments:
        average_surprise_counts[segment] += surprise_counts[segment]
        average_novelty_counts[segment] += novelty_counts[segment]

# Calculate the averages
for segment in segments:
    average_surprise_counts[segment] /= n_runs
    average_novelty_counts[segment] /= n_runs

# List of trial segments
segments_first = ['50', '100', '200', '500']  # First set of segments
segments_second = ['1000', '2000', '5000', '10000']  # Second set of segments

states = np.arange(11)  # Assuming there are 11 states, adjust this as needed

# Colors for states (same color for the states across both agents)
state_colors = sns.color_palette("muted", len(states))

# --- First Plot: Segments 50 to 500 ---

fig, ax1 = plt.subplots(figsize=(10, 6))

# Positions for the trial segments on the x-axis
bar_width = 0.4
r1 = np.arange(len(segments_first))
r2 = [x + bar_width for x in r1]

# Loop through each trial segment and create stacked bars
for i, segment in enumerate(segments_first):
    surprise_data = average_surprise_counts[segment]
    novelty_data = average_novelty_counts[segment]

    # Stacked bar for surprise agent
    bottom = np.zeros(len(segments_first))
    for state_idx, color in enumerate(state_colors):
        ax1.bar(r1[i], surprise_data[state_idx], color=color, width=bar_width, edgecolor='grey',
                label=f'State {state_idx}' if i == 0 else "", bottom=bottom[i])
        bottom[i] += surprise_data[state_idx]

    # Stacked bar for novelty agent
    bottom = np.zeros(len(segments_first))
    for state_idx, color in enumerate(state_colors):
        ax1.bar(r2[i], novelty_data[state_idx], color=color, width=bar_width, edgecolor='grey',
                bottom=bottom[i])
        bottom[i] += novelty_data[state_idx]

# Add labels for agents
for i in range(len(segments_first)):
    ax1.text(r1[i], -700, 'Surprise', ha='center', rotation=90, va='center', color='black', rotation_mode="anchor")
    ax1.text(r2[i], -700, 'Novelty', ha='center', rotation=90, va='center', color='black', rotation_mode="anchor")

# Add labels and title for the first plot
ax1.set_xlabel('Trial Segments (50 to 500)', fontweight='bold')
ax1.set_ylabel('State Visits', fontweight='bold')
ax1.set_title(f'State Visits per Trial Segment (50 to 500), averaged over {n_runs} agents', fontweight='bold')

# Custom x-axis labels
ax1.set_xticks([r + bar_width / 2 for r in range(len(segments_first))])
ax1.set_xticklabels(segments_first)

# Show legend for the states (only display once)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[:len(states)], labels[:len(states)], title="States")

# --- Second Plot: Segments 1000 to 10000 ---

fig, ax2 = plt.subplots(figsize=(10, 6))

# Positions for the trial segments on the x-axis
r1 = np.arange(len(segments_second))
r2 = [x + bar_width for x in r1]

# Loop through each trial segment and create stacked bars
for i, segment in enumerate(segments_second):
    surprise_data = average_surprise_counts[segment]
    novelty_data = average_novelty_counts[segment]

    # Stacked bar for surprise agent
    bottom = np.zeros(len(segments_second))
    for state_idx, color in enumerate(state_colors):
        ax2.bar(r1[i], surprise_data[state_idx], color=color, width=bar_width, edgecolor='grey',
                label=f'State {state_idx}' if i == 0 else "", bottom=bottom[i])
        bottom[i] += surprise_data[state_idx]

    # Stacked bar for novelty agent
    bottom = np.zeros(len(segments_second))
    for state_idx, color in enumerate(state_colors):
        ax2.bar(r2[i], novelty_data[state_idx], color=color, width=bar_width, edgecolor='grey',
                bottom=bottom[i])
        bottom[i] += novelty_data[state_idx]

# Add labels for agents
for i in range(len(segments_second)):
    ax2.text(r1[i], -700, 'Surprise', ha='center', rotation=90, va='center', color='black', rotation_mode="anchor")
    ax2.text(r2[i], -700, 'Novelty', ha='center', rotation=90, va='center', color='black', rotation_mode="anchor")

# Add labels and title for the second plot
ax2.set_xlabel('Trial Segments (1000 to 10000)', fontweight='bold')
ax2.set_ylabel('State Visits', fontweight='bold')
ax2.set_title(f'State Visits per Trial Segment (1000 to 10000), averaged over {n_runs} agents', fontweight='bold')

# Custom x-axis labels
ax2.set_xticks([r + bar_width / 2 for r in range(len(segments_second))])
ax2.set_xticklabels(segments_second)

# Show legend for the states (only display once)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[:len(states)], labels[:len(states)], title="States")

# Show the plots
plt.tight_layout()
plt.show()