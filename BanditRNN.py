import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from scipy import interpolate

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.optimize import curve_fit




# Define two armed bandit environment
class KalmanBandit:
    def __init__(self, n_arms=2, n_states=100, innov_variance = 100):
        self.n_arms = n_arms
        self.noise_variance = 10
        self.post_mean = np.zeros(n_arms)
        self.post_mean_array = np.zeros((n_arms, n_states))
        self.post_variance = np.ones(n_arms)*10
        self.post_variance_array = np.zeros((n_arms, n_states))
        self.kalman_gain = np.zeros((n_arms, n_states))
        #self.reward = np.random.normal(0, 1, (n_arms, n_states))
        #self.ucb = np.zeros((n_arms, n_states))
        self.V_t = np.zeros(n_states)
        self.std_dev = np.zeros(n_states)
        self.P_a0_thompson = np.zeros(n_states)
        self.P_a0_ucb = np.zeros(n_states)
        self.innov_variance = innov_variance
        self.beta = 1.96
        self.gamma = 1
        self.theta = 0.8

    def update(self, action, state, reward):
        # TODO: implement Kalman filter exactly as they do in the paper
        for i in range(self.n_arms):
            if action == i:
                self.kalman_gain[i][state] = self.post_variance[i] / (self.post_variance[i] +
                                                                     self.noise_variance)
                                #self.kalman_gain[i][state] = (self.post_variance[i] + self.innov_variance) / (self.post_variance[i] +
                                                                                     #self.innov_variance + self.noise_variance)
            else:
                self.kalman_gain[i][state] = 0
            self.post_variance[i] = self.post_variance[i] - self.kalman_gain[i][state] * self.post_variance[i] #(1 - self.kalman_gain[i][state]) * (self.post_variance[i] + self.innov_variance)
        self.post_mean[action] = self.post_mean[action] + self.kalman_gain[action][state] * (reward - self.post_mean[action])
        for i in range(self.n_arms):
            # append to the arrays after each update
            self.post_mean_array[i][state] = self.post_mean[i]
            self.post_variance_array[i][state] = self.post_variance[i]



    def ucb(self, state):
        self.V_t[state] = self.post_mean[0] - self.post_mean[1]  # Example difference in means
        sigma2_1 = np.sqrt(self.post_variance[0])
        sigma2_2 = np.sqrt(self.post_variance[1])
        self.P_a0_ucb[state] = norm.cdf((self.V_t[state] + self.gamma * (sigma2_1 - sigma2_2)) / self.theta)
        # sample action 0 with probability P_a0_ucb
        action = 0 if np.random.rand() < self.P_a0_ucb[state] else 1


        #self.ucb[0][state] = self.post_mean[0][state-1] + self.beta * (np.sqrt(self.post_variance[0] + self.innov_variance))
        #self.ucb[1][state] = self.post_mean[1][state-1] + self.beta * (np.sqrt(self.post_variance[1] + self.innov_variance))

        return action

    def thompson(self, state):
        # modelling
        # Define values for V_t, sigma^2_t(1), and sigma^2_t(2)
        self.V_t[state] = self.post_mean[0] - self.post_mean[1]  # Example difference in means
        sigma2_1 = self.post_variance[0]  # Variance of arm 1
        sigma2_2 = self.post_variance[1]   # Variance of arm 2

        # Compute the standard deviation for the combined variance
        self.std_dev = np.sqrt(sigma2_1 + sigma2_2)

        # Calculate the probability P(a_t = 1)
        self.P_a0_thompson[state] = norm.cdf((self.V_t[state] / self.std_dev))
        action = 0 if np.random.rand() < self.P_a0_thompson[state] else 1

        return action

n_participants = 30
n_block_per_p = 20
n_trials_per_block = 10
reward_array = np.empty([2, n_trials_per_block * n_block_per_p * n_participants], dtype=float)
data_ucb = []
data_thompson = []
algorithms = ["ucb", "thompson"]

for algorithm in algorithms:
    state = 0
    for participant in range(n_participants):
        agent = KalmanBandit(n_states=n_participants * n_block_per_p * n_trials_per_block)
        for block in range(n_block_per_p):
            mean_reward_block = np.random.normal(0, np.sqrt(agent.innov_variance), agent.n_arms)
            for trial in range(n_trials_per_block):
                # Define the environment
                reward = np.random.normal(mean_reward_block, np.sqrt(agent.noise_variance), agent.n_arms)


                # Run the agent in the environment
                if algorithm == "thompson":
                    action = agent.thompson(state)
                else:
                    action = agent.ucb(state)
                reward = reward[action]
                agent.update(action, state = state, reward = reward)


                if algorithm == "thompson":
                    data_thompson.append({
                        'Participant': participant,
                        'Block': block,
                        'Trial': trial,
                        'State': state,
                        'Action': action,
                        'Reward': reward,
                        'P_a0_thompson': agent.P_a0_thompson[state],
                        'V_t': agent.V_t[state],
                        'posterior_std_0': agent.post_variance[0],
                        'posterior_std_1': agent.post_variance[1],
                        'poster_mean_0': agent.post_mean[0],
                        'poster_mean_1': agent.post_mean[1],
                        'TU': np.sqrt(agent.post_variance[0] + agent.post_variance[1]),
                        'RU': np.sqrt(agent.post_variance[0]) - np.sqrt(agent.post_variance[1])

                    })
                else:

                    data_ucb.append({
                        'Participant': participant,
                        'Block': block,
                        'Trial': trial,
                        'State': state,
                        'Action': action,
                        'Reward': reward,
                        'P_a0_ucb': agent.P_a0_ucb[state],
                        'V_t': agent.V_t[state],
                        'posterior_std_0': agent.post_variance[0],
                        'posterior_std_1': agent.post_variance[1],
                        'poster_mean_0': agent.post_mean[0],
                        'poster_mean_1': agent.post_mean[1],
                        'TU': np.sqrt(agent.post_variance[0] + agent.post_variance[1]),
                        'RU': np.sqrt(agent.post_variance[0]) - np.sqrt(agent.post_variance[1])

                    })
                    state += 1

df_ucb = pd.DataFrame(data_ucb)
df_thompson = pd.DataFrame(data_thompson)

# Define the directory and file path
output_dir = "data"
output_file_ucb = "results_ucb.csv"
output_file_thompson = "results_thompson.csv"
output_file_grouped_ucb = "results_grouped_ucb.csv"
output_file_grouped_thompson = "results_grouped_thompson.csv"



# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save the DataFrame to a CSV file in the specified directory
df_ucb.to_csv(os.path.join(output_dir, output_file_ucb), index=False)
df_thompson.to_csv(os.path.join(output_dir, output_file_thompson), index=False)




# plotting in R for now
### plotting

# Step 1: Create "Low TU" and "High TU" groups based on the median of `posterior_std_0`
#ucb
"""v_bins = np.linspace(df_ucb['V_t'].min(), df_ucb['V_t'].max(), 8)
df_ucb['V_binned'] = pd.cut(df_ucb['V_t'], bins=v_bins)"""

"""median_std = df_ucb['TU'].median()
df_ucb['TU_group'] = df_ucb['TU'].apply(
    lambda x: 'Low TU' if x <= median_std else 'High TU')"""

#thompson
"""median_thompson = df_thompson['TU'].median()
df_thompson['TU_group'] = df_thompson['TU'].apply(
    lambda x: 'Low TU' if x <= median_std else 'High TU')"""

# Step 2: Group data by "TU_group" and `V_t`, then calculate mean choice probability (`P_a0_ucb`)
#ucb
"""grouped_data = df_ucb.groupby(['TU_group', 'V_binned']).agg(
    mean_choice_prob=('P_a0_ucb', 'mean')).reset_index()
grouped_data.to_csv(os.path.join(output_dir, output_file_grouped_ucb), index=False)
"""
"""#thompson
grouped_data = df_thompson.groupby(['TU_group', 'V_t']).agg(
    mean_choice_prob=('P_a0_thompson', 'mean')).reset_index()
grouped_data.to_csv(os.path.join(output_dir, output_file_grouped_thompson), index=False)"""


"""# Step 3: Plot the data
plt.figure(figsize=(6, 4))

# Plot each group separately
sns.lineplot(
    data=grouped_data[grouped_data['TU_group'] == 'Low TU'],
    x='V_binned', y='mean_choice_prob', marker='o', color='black', label='Low TU'
)
sns.lineplot(
    data=grouped_data[grouped_data['TU_group'] == 'High TU'],
    x='V_binned', y='mean_choice_prob', marker='o', color='gray', label='High TU'
)

# Set plot labels and title
plt.xlabel('Expected value difference')
plt.ylabel('Choice probability')
plt.title('Choice Probability by Expected Value Difference and TU Group')
plt.legend(title=None)
plt.xlim(-10, 10)  # Adjust x-axis limits for a similar look to the screenshot
plt.ylim(-0.1, 1.1)  # Set y-axis from 0 to 1
plt.grid(True)
plt.show()"""

## calculate mean choice probability for each 1 point step in expected value difference and plot this

"""# make the plot smoother
# Sort data by `V_t` for a clear trend
#ucb
df_ucb_sorted = df_ucb.sort_values(by='V_t')

#thompson
df_thompson_sorted = df_thompson.sort_values(by='V_t')


# Apply smoothing (rolling mean) separately for each TU group
#ucb
df_ucb_sorted['Smoothed_P_a0_ucb'] = df_ucb_sorted.groupby('TU_group')['P_a0_ucb'].transform(lambda x: x.rolling(window=20, center=True).mean())
#thompson
df_thompson_sorted['Smoothed_P_a0_thompson'] = df_thompson_sorted.groupby('TU_group')['P_a0_thompson'].transform(lambda x: x.rolling(window=20, center=True).mean())
"""

"""# Step 3: Plot the smoothed data - ucb
plt.figure(figsize=(6, 4))

# Plot each group with the smoothed values
sns.lineplot(
    data=df_ucb_sorted[df_ucb_sorted['TU_group'] == 'Low TU'],
    x='V_t', y='Smoothed_P_a0_ucb', marker='o', color='black', label='Low TU'
)
sns.lineplot(
    data=df_ucb_sorted[df_ucb_sorted['TU_group'] == 'High TU'],
    x='V_t', y='Smoothed_P_a0_ucb', marker='o', color='gray', label='High TU'
)
# Set plot labels and title
plt.xlabel('Expected value difference')
plt.ylabel('Choice probability')
plt.title('UCB - Choice Probability by Expected Value Difference and TU Group (Smoothed)')
plt.legend(title=None)
plt.xlim(-10, 10)  # Adjust x-axis limits if needed
plt.ylim(-0.1, 1.1)   # Set y-axis from 0 to 1
plt.grid(True)
plt.show()"""

"""# Step 3: Plot the smoothed data - thompson
plt.figure(figsize=(6, 4))

# Plot each group with the smoothed values
sns.lineplot(
    data=df_thompson_sorted[df_thompson_sorted['TU_group'] == 'Low TU'],
    x='V_t', y='Smoothed_P_a0_thompson', marker='o', color='black', label='Low TU'
)
sns.lineplot(
    data=df_thompson_sorted[df_thompson_sorted['TU_group'] == 'High TU'],
    x='V_t', y='Smoothed_P_a0_thompson', marker='o', color='gray', label='High TU'
)
# Set plot labels and title
plt.xlabel('Expected value difference')
plt.ylabel('Choice probability')
plt.title('Thompson - Choice Probability by Expected Value Difference and TU Group (Smoothed)')
plt.legend(title=None)
plt.xlim(-10, 10)  # Adjust x-axis limits if needed
plt.ylim(-0.1, 1.1)   # Set y-axis from 0 to 1
plt.grid(True)
plt.show()"""


"""# Calculate the 5th percentile cutoff
percentile_5 = df_ucb['posterior_std_0'].quantile(0.10)

# Filter the DataFrame for values in the lowest 5 percentile
filtered_df = df_ucb[df_ucb['posterior_std_0'] <= percentile_5]

# Plot the filtered data
plt.figure(figsize=(12, 6))
sns.lineplot(data=filtered_df, x='V_t', y='P_a0_ucb', ci=None)
plt.xlabel('Expected Value Difference')
plt.ylabel('Choice Probability')
plt.title('Choice Probability for Action 0 (UCB) - Lowest 10% Posterior Std')
plt.show()"""