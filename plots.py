import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time


def plot_stacked_individual_segments(stoch, segments, average_surprise_counts, average_novelty_counts, n_runs, output_dir):


    # Colors for each group
    if not stoch:
        color_groups = {
            'group_1_3': sns.color_palette("muted", 4)[0],  # Color for states 1-3
            'group_4_6': sns.color_palette("muted", 4)[1],  # Color for states 4-6
            'TrapStates': sns.color_palette("muted", 4)[2],  # Color for states 7-9
            'random': sns.color_palette("muted", 4)[3]  # Color for random states
        }
    else:
        color_groups = {
            'progress': sns.color_palette("muted", 3)[0],  # Color for states 1-3
            'trap': sns.color_palette("muted", 3)[1],  # Color for states 4-6
            'stochastic': sns.color_palette("muted", 3)[2],  # Color for states 7-9
        }

    # Loop through each segment and plot separately
    for segment in segments:
        fig, ax = plt.subplots(figsize=(8, 6))

        surprise_data = average_surprise_counts[segment]
        novelty_data = average_novelty_counts[segment]

        # Bar width
        bar_width = 0.0005
        r1 = np.arange(1)  # Single bar for surprise agent
        r2 = [x + bar_width for x in r1]  # Single bar for novelty agent

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

            # Stacked bar for surprise agent
            ax.bar(r1, surprise_group_1_3, color=color_groups['group_1_3'], width=bar_width, edgecolor='grey',
                   label='States 1 and 2')
            ax.bar(r1, surprise_random_group, bottom=surprise_group_1_3, color=color_groups['random'], width=bar_width,
                   edgecolor='grey', label='Stochastic States')
            ax.bar(r1, surprise_group_4_6, bottom=surprise_random_group + surprise_group_1_3,
                   color=color_groups['group_4_6'], width=bar_width,
                   edgecolor='grey', label='States 5 and 6')
            ax.bar(r1, surprise_group_7_9, bottom=surprise_group_1_3 + surprise_group_4_6 + surprise_random_group,
                   color=color_groups['TrapStates'],
                   width=bar_width, edgecolor='grey', label='Trap States')

            # Stacked bar for novelty agent
            ax.bar(r2, novelty_group_1_3, color=color_groups['group_1_3'], width=bar_width, edgecolor='grey')
            ax.bar(r2, novelty_random_group, bottom=novelty_group_1_3, color=color_groups['random'], width=bar_width,
                   edgecolor='grey')
            ax.bar(r2, novelty_group_4_6, bottom=novelty_random_group + novelty_group_1_3, color=color_groups['group_4_6'],
                   width=bar_width,
                   edgecolor='grey')
            ax.bar(r2, novelty_group_7_9, bottom=novelty_group_1_3 + novelty_group_4_6 + novelty_random_group,
                   color=color_groups['group_7_9'],
                   width=bar_width, edgecolor='grey')
        else:
            surprise_progress = np.sum(surprise_data[0:6])  # Sum states 1, 2, 3
            surprise_trap = np.sum(surprise_data[6:8])
            surprise_stoch = np.sum(surprise_data[11:61])  # Sum states 4, 5, 6

            novelty_progress = np.sum(novelty_data[0:6])
            novelty_trap = np.sum(novelty_data[6:8])
            novelty_stoch = np.sum(novelty_data[11:61])

            # Stacked bar for surprise agent
            ax.bar(r1, surprise_progress, color=color_groups['progress'], width=bar_width, edgecolor='grey',
                   label='Progress States')
            ax.bar(r1, surprise_trap, bottom=surprise_progress, color=color_groups['trap'], width=bar_width,
                   edgecolor='grey', label='Trap States')
            ax.bar(r1, surprise_stoch, bottom=surprise_progress+ surprise_trap,
                   color=color_groups['stochastic'], width=bar_width,
                   edgecolor='grey', label='Stochastic States')


            # Stacked bar for novelty agent
            ax.bar(r2, novelty_progress, color=color_groups['progress'], width=bar_width, edgecolor='grey')
            ax.bar(r2, novelty_trap, bottom=novelty_progress, color=color_groups['trap'], width=bar_width,
                   edgecolor='grey')
            ax.bar(r2, novelty_stoch, bottom=novelty_progress+ novelty_trap,
                   color=color_groups['stochastic'], width=bar_width,
                   edgecolor='grey')



        # Add labels for agents
        ax.text(r1[0], -0.01*int(segment), 'Surprise', ha='center', va='center', color='black')
        ax.text(r2[0], -0.01*int(segment), 'Novelty', ha='center', va='center', color='black')

        # Set labels and title
        ax.set_ylabel('State Visits', fontweight='bold')
        ax.set_title(f'State Visits in Segment {segment}', fontweight='bold')

        # Custom x-axis labels
        ax.set_xticks([r + bar_width / 2 for r in range(1)])
        ax.set_xticklabels(['Agents'])

        # Show legend for the state groups
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:3], labels[:3], title="State Groups")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(output_dir, f'stoch_segment_{segment}_{timestamp}.png')

        plt.savefig(filename, format='png')
        print(f'Saved plot for segment {segment} as {filename}')
        plt.clf()
    return plt.show()


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