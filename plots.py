import os

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import matplotlib.cm as cm
import matplotlib as mpl
import ast




def plot_stacked_individual_segments(average_surprise_counts, average_novelty_counts, average_random_counts, segments, stoch, output_dir, n_runs):
    # Colors for each group
    if not stoch:
        color_groups = {
            'group_1_3': sns.color_palette("muted", 4)[0],  # Color for states 1-3
            'group_4_6': sns.color_palette("muted", 4)[1],  # Color for states 4-6
            'TrapStates': sns.color_palette("muted", 4)[2],  # Color for states 7-9
            'random': sns.color_palette("muted", 4)[3]      # Color for random states
        }
    else:
        color_groups = {
            'progress': sns.color_palette("muted", 3)[0],   # Color for states 1-3
            'trap': sns.color_palette("muted", 3)[1],       # Color for states 4-6
            'stochastic': sns.color_palette("muted", 3)[2], # Color for stochastic states
        }

    # Loop through each segment and plot separately
    for segment in segments:
        fig, ax = plt.subplots(figsize=(10, 6))

        surprise_data = average_surprise_counts[segment]
        novelty_data = average_novelty_counts[segment]
        random_data = average_random_counts[segment]  # Data for random agent

        # Bar width
        bar_width = 0.0005
        r1 = np.arange(1)  # Single bar for surprise agent
        r2 = [x + bar_width for x in r1]  # Single bar for novelty agent
        r3 = [x + bar_width * 2 for x in r1]  # Single bar for random agent

        if not stoch:
            # Group states together
            surprise_group_1_3 = np.sum(surprise_data[0:2])  # Sum states 1, 2, 3
            surprise_random_group = np.sum(surprise_data[2:4])
            surprise_group_4_6 = np.sum(surprise_data[4:7])  # Sum states 4, 5, 6
            surprise_group_7_9 = np.sum(surprise_data[7:10])  # Sum states 7, 8, 9

            novelty_group_1_3 = np.sum(novelty_data[0:2])
            novelty_random_group = np.sum(novelty_data[2:4])
            novelty_group_4_6 = np.sum(novelty_data[4:7])
            novelty_group_7_9 = np.sum(novelty_data[7:10])

            random_group_1_3 = np.sum(random_data[0:2])
            random_random_group = np.sum(random_data[2:4])
            random_group_4_6 = np.sum(random_data[4:7])
            random_group_7_9 = np.sum(random_data[7:10])

            # Stacked bar for surprise agent
            ax.bar(r1, surprise_group_1_3, color=color_groups['group_1_3'], width=bar_width, edgecolor='grey',
                   label='States 1 and 2')
            ax.bar(r1, surprise_random_group, bottom=surprise_group_1_3, color=color_groups['random'], width=bar_width,
                   edgecolor='grey', label='Stochastic States')
            ax.bar(r1, surprise_group_4_6, bottom=surprise_group_1_3 + surprise_random_group,
                   color=color_groups['group_4_6'], width=bar_width, edgecolor='grey', label='States 5 and 6')
            ax.bar(r1, surprise_group_7_9, bottom=surprise_group_1_3 + surprise_group_4_6 + surprise_random_group,
                   color=color_groups['TrapStates'], width=bar_width, edgecolor='grey', label='Trap States')

            # Stacked bar for novelty agent
            ax.bar(r2, novelty_group_1_3, color=color_groups['group_1_3'], width=bar_width, edgecolor='grey')
            ax.bar(r2, novelty_random_group, bottom=novelty_group_1_3, color=color_groups['random'], width=bar_width,
                   edgecolor='grey')
            ax.bar(r2, novelty_group_4_6, bottom=novelty_random_group + novelty_group_1_3, color=color_groups['group_4_6'],
                   width=bar_width, edgecolor='grey')
            ax.bar(r2, novelty_group_7_9, bottom=novelty_group_1_3 + novelty_group_4_6 + novelty_random_group,
                   color=color_groups['TrapStates'], width=bar_width, edgecolor='grey')

            # Stacked bar for random agent
            ax.bar(r3, random_group_1_3, color=color_groups['group_1_3'], width=bar_width, edgecolor='grey')
            ax.bar(r3, random_random_group, bottom=random_group_1_3, color=color_groups['random'], width=bar_width,
                   edgecolor='grey')
            ax.bar(r3, random_group_4_6, bottom=random_random_group + random_group_1_3, color=color_groups['group_4_6'],
                   width=bar_width, edgecolor='grey')
            ax.bar(r3, random_group_7_9, bottom=random_group_1_3 + random_group_4_6 + random_random_group,
                   color=color_groups['TrapStates'], width=bar_width, edgecolor='grey')

        else:
            surprise_progress = np.sum(surprise_data[0, 0:6])  # Sum states 1, 2, 3
            surprise_trap = np.sum(surprise_data[0, 6:8])
            surprise_stoch = np.sum(surprise_data[0, 11:61])  # Sum states 4, 5, 6

            novelty_progress = np.sum(novelty_data[0, 0:6])
            novelty_trap = np.sum(novelty_data[0, 6:8])
            novelty_stoch = np.sum(novelty_data[0, 11:61])

            random_progress = np.sum(random_data[0, 0:6])
            random_trap = np.sum(random_data[0, 6:8])
            random_stoch = np.sum(random_data[0, 11:61])

            # Stacked bar for surprise agent
            ax.bar(r1, surprise_progress, color=color_groups['progress'], width=bar_width, edgecolor='grey',
                   label='Progress States')
            ax.bar(r1, surprise_trap, bottom=surprise_progress, color=color_groups['trap'], width=bar_width,
                   edgecolor='grey', label='Trap States')
            ax.bar(r1, surprise_stoch, bottom=surprise_progress + surprise_trap,
                   color=color_groups['stochastic'], width=bar_width, edgecolor='grey', label='Stochastic States')

            # Stacked bar for novelty agent
            ax.bar(r2, novelty_progress, color=color_groups['progress'], width=bar_width, edgecolor='grey')
            ax.bar(r2, novelty_trap, bottom=novelty_progress, color=color_groups['trap'], width=bar_width,
                   edgecolor='grey')
            ax.bar(r2, novelty_stoch, bottom=novelty_progress + novelty_trap,
                   color=color_groups['stochastic'], width=bar_width, edgecolor='grey')

            # Stacked bar for random agent
            ax.bar(r3, random_progress, color=color_groups['progress'], width=bar_width, edgecolor='grey')
            ax.bar(r3, random_trap, bottom=random_progress, color=color_groups['trap'], width=bar_width,
                   edgecolor='grey')
            ax.bar(r3, random_stoch, bottom=random_progress + random_trap,
                   color=color_groups['stochastic'], width=bar_width, edgecolor='grey')



        # Set labels and title
        ax.set_ylabel('State Visits', fontweight='bold')
        ax.set_title(f'State Visits in Segment {segment}', fontweight='bold')

        # Custom x-axis labels
        ax.set_xticks([r1[0], r2[0], r3[0]])  # Set the positions to match the actual bars
        ax.set_xticklabels(['Surprise', 'Novelty', 'Random'])  # Set labels for each agent
        ax.set_xlabel(f"Averaged over {n_runs} agents")
        # Show legend for the state groups
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:3], labels[:3], title="State Groups")

        # Save the plot
        filename = os.path.join(output_dir, f'stoch_segment_{segment}.png')
        plt.savefig(filename, format='png')
        print(f'Saved plot for segment {segment} as {filename}')
        plt.clf()


def save_ratios_to_csv(overall_random_counts_ratio, overall_novelty_counts_ratio, overall_surprise_counts_ratio,
                       segments_ratio, n_runs, filename='/Users/marvinmathony/Documents/RLandNNs/data/agent_ratios.csv'):
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Append the timestamp to the filename
    full_filename = f"{filename}_{timestamp}.csv"

    # Create a dictionary where each key is a column in the CSV
    data = {
        'Segment': segments_ratio,
        'Random_Agent': [overall_random_counts_ratio[segment] for segment in segments_ratio],
        'Novelty_Agent': [overall_novelty_counts_ratio[segment] for segment in segments_ratio],
        'Surprise_Agent': [overall_surprise_counts_ratio[segment] for segment in segments_ratio],
        'Runs': [n_runs] * len(segments_ratio)  # Add n_runs information in a column
    }

    # Convert to a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(full_filename, index=False)
    print(f"Data saved to {full_filename}")
    return full_filename


def load_ratios_from_csv(filename):
    # Read the CSV into a DataFrame
    df = pd.read_csv(filename)

    # Initialize the dictionaries
    overall_random_counts_ratio = {}
    overall_novelty_counts_ratio = {}
    overall_surprise_counts_ratio = {}

    # Populate the dictionaries
    for _, row in df.iterrows():
        segment = row['Segment']
        overall_random_counts_ratio[segment] = numpy.fromstring(row['Random_Agent'], sep = ",")
        overall_novelty_counts_ratio[segment] = numpy.fromstring(row['Novelty_Agent'], sep = ",")
        overall_surprise_counts_ratio[segment] = numpy.fromstring(row['Surprise_Agent'], sep=",")

    return list(df['Segment']), overall_random_counts_ratio, overall_novelty_counts_ratio, overall_surprise_counts_ratio

def compute_ratios(step_count):
    print(step_count)
    trap_states = step_count[0, 6:8]
    stochastic_states = step_count[0, 11:61]


    total_visits = np.sum(step_count)
    trap_visits = np.sum(trap_states)
    stochastic_visits = np.sum(stochastic_states)

    trap_ratio = trap_visits / total_visits
    stochastic_ratio = stochastic_visits / total_visits
    progressing_ratio = 1 - (trap_ratio + stochastic_ratio)
    return trap_ratio, stochastic_ratio, progressing_ratio

def plot_ratios(segments, overall_random_counts, overall_novelty_counts, overall_surprise_counts):
    trap_ratio_random, stochastic_ratio_random, progressing_ratio_random = [], [], []
    trap_ratio_surprise, stochastic_ratio_surprise, progressing_ratio_surprise = [], [], []
    trap_ratio_novelty, stochastic_ratio_novelty, progressing_ratio_novelty = [], [], []

    for segment in segments:
        # Get state visits for each agent
        state_visits_random = overall_random_counts[segment]
        state_visits_surprise = overall_surprise_counts[segment]
        state_visits_novelty = overall_novelty_counts[segment]

        # Random agent ratios
        print(f"type of state_visits_random: {type(state_visits_random)}")
        trap, stochastic, progressing = compute_ratios(state_visits_random)
        trap_ratio_random.append(trap)
        stochastic_ratio_random.append(stochastic)
        progressing_ratio_random.append(progressing)

        # Surprise agent ratios
        trap, stochastic, progressing = compute_ratios(state_visits_surprise)
        trap_ratio_surprise.append(trap)
        stochastic_ratio_surprise.append(stochastic)
        progressing_ratio_surprise.append(progressing)

        # Novelty agent ratios
        trap, stochastic, progressing = compute_ratios(state_visits_novelty)
        trap_ratio_novelty.append(trap)
        stochastic_ratio_novelty.append(stochastic)
        progressing_ratio_novelty.append(progressing)

    # Convert segments to a numpy array for colormap scaling
    segments_array = np.array(segments).astype(float)
    norm = plt.Normalize(segments_array.min(), segments_array.max())
    cmap = mpl.colormaps['viridis']  # You can change the colormap if you prefer

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Plot random agent (blue)
    sc = plt.scatter(stochastic_ratio_random, trap_ratio_random, c=segments_array, cmap=cmap, label='Random', marker='o')
    plt.plot(stochastic_ratio_random, trap_ratio_random, color='b', label='Random Line')

    # Plot surprise agent (green)
    plt.scatter(stochastic_ratio_surprise, trap_ratio_surprise, c=segments_array, cmap=cmap, label='Surprise', marker='s')
    plt.plot(stochastic_ratio_surprise, trap_ratio_surprise, color='g', label='Surprise Line')

    # Plot novelty agent (red)
    plt.scatter(stochastic_ratio_novelty, trap_ratio_novelty, c=segments_array, cmap=cmap, label='Novelty', marker='^')
    plt.plot(stochastic_ratio_novelty, trap_ratio_novelty, color='r', label='Novelty Line')

    # Add colorbar to indicate steps
    cbar = plt.colorbar(sc)
    cbar.set_label('Steps (Segments)')

    # Label the axes
    plt.xlabel('Ratio of Stochastic States')
    plt.ylabel('Ratio of Trap States')
    plt.title('State Visit Ratios: Trap vs Stochastic')

    # Add legend
    plt.legend()

    # Add grid for readability
    plt.grid(True)

    # Show the plot
    plt.show()


def plot_ratios_random(segments, overall_random_counts, overall_surprise_counts, overall_novelty_counts):
    trap_ratio_random, stochastic_ratio_random, progressing_ratio_random = [], [], []
    trap_ratio_surprise, stochastic_ratio_surprise, progressing_ratio_surprise = [], [], []
    trap_ratio_novelty, stochastic_ratio_novelty, progressing_ratio_novelty = [], [], []

    for segment in segments:
        # Get state visits for each agent
        state_visits_random = overall_random_counts[segment]
        state_visits_surprise = overall_surprise_counts[segment]
        state_visits_novelty = overall_novelty_counts[segment]

        # Random agent ratios
        print(f"type of state_visits_random: {type(state_visits_random)}")
        trap, stochastic, progressing = compute_ratios(state_visits_random)
        trap_ratio_random.append(trap)
        stochastic_ratio_random.append(stochastic)
        progressing_ratio_random.append(progressing)

        # Surprise agent ratios
        trap, stochastic, progressing = compute_ratios(state_visits_surprise)
        trap_ratio_surprise.append(trap)
        stochastic_ratio_surprise.append(stochastic)
        progressing_ratio_surprise.append(progressing)

        # Novelty agent ratios
        trap, stochastic, progressing = compute_ratios(state_visits_novelty)
        trap_ratio_novelty.append(trap)
        stochastic_ratio_novelty.append(stochastic)
        progressing_ratio_novelty.append(progressing)

    # Convert segments to a numpy array for colormap scaling
    segments_array = np.array(segments).astype(float)
    norm = plt.Normalize(segments_array.min(), segments_array.max())
    cmap = mpl.colormaps['viridis']  # You can change the colormap if you prefer

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Plot random agent (blue)
    sc = plt.scatter(stochastic_ratio_random, trap_ratio_random, c=segments_array, cmap=cmap, label='Random', marker='o')
    plt.plot(stochastic_ratio_random, trap_ratio_random, color='b', label='Random Line')

    # Add colorbar to indicate steps
    cbar = plt.colorbar(sc)
    cbar.set_label('Steps (Segments)')

    # Label the axes
    plt.xlabel('Ratio of Stochastic States')
    plt.ylabel('Ratio of Trap States')
    plt.title('State Visit Ratios: Trap vs Stochastic')

    # Add legend
    plt.legend()

    # Add grid for readability
    plt.grid(True)

    # Show the plot
    plt.show()

def plot_ratios_random_surprise(segments, overall_random_counts, overall_surprise_counts, overall_novelty_counts):
    trap_ratio_random, stochastic_ratio_random, progressing_ratio_random = [], [], []
    trap_ratio_surprise, stochastic_ratio_surprise, progressing_ratio_surprise = [], [], []
    trap_ratio_novelty, stochastic_ratio_novelty, progressing_ratio_novelty = [], [], []

    for segment in segments:
        # Get state visits for each agent
        state_visits_random = overall_random_counts[segment]
        state_visits_surprise = overall_surprise_counts[segment]
        state_visits_novelty = overall_novelty_counts[segment]

        # Random agent ratios
        print(f"type of state_visits_random: {type(state_visits_random)}")
        trap, stochastic, progressing = compute_ratios(state_visits_random)
        trap_ratio_random.append(trap)
        stochastic_ratio_random.append(stochastic)
        progressing_ratio_random.append(progressing)

        # Surprise agent ratios
        trap, stochastic, progressing = compute_ratios(state_visits_surprise)
        trap_ratio_surprise.append(trap)
        stochastic_ratio_surprise.append(stochastic)
        progressing_ratio_surprise.append(progressing)

        # Novelty agent ratios
        trap, stochastic, progressing = compute_ratios(state_visits_novelty)
        trap_ratio_novelty.append(trap)
        stochastic_ratio_novelty.append(stochastic)
        progressing_ratio_novelty.append(progressing)

    # Convert segments to a numpy array for colormap scaling
    segments_array = np.array(segments).astype(float)
    norm = plt.Normalize(segments_array.min(), segments_array.max())
    cmap = mpl.colormaps['viridis']  # You can change the colormap if you prefer

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Plot random agent (blue)
    sc = plt.scatter(stochastic_ratio_random, trap_ratio_random, c=segments_array, cmap=cmap, label='Random', marker='o')
    plt.plot(stochastic_ratio_random, trap_ratio_random, color='b', label='Random Line')

    # Plot surprise agent (green)
    plt.scatter(stochastic_ratio_surprise, trap_ratio_surprise, c=segments_array, cmap=cmap, label='Surprise', marker='s')
    plt.plot(stochastic_ratio_surprise, trap_ratio_surprise, color='g', label='Surprise Line')



    # Add colorbar to indicate steps
    cbar = plt.colorbar(sc)
    cbar.set_label('Steps (Segments)')

    # Label the axes
    plt.xlabel('Ratio of Stochastic States')
    plt.ylabel('Ratio of Trap States')
    plt.title('State Visit Ratios: Trap vs Stochastic')

    # Add legend
    plt.legend()

    # Add grid for readability
    plt.grid(True)

    # Show the plot
    plt.show()






def plot_stacked_individual_bars_two_images(n_runs, ):

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
    return plt.tight_layout(), plt.show()