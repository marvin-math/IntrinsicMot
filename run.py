import gym
import statistics
import envs.environment

env = gym.make('MDPAlireza-v0')

# Number of episodes for the agent to run
num_episodes = 100
list_of_steps = []

for episode in range(num_episodes):
    # Reset the environment for each episode
    observation, info = env.reset()
    terminated = False
    step_count = 0
    total_reward = 0

    print(f"Episode {episode + 1} started:")

    # Run the environment until termination or max steps
    while not terminated and step_count < 300000:  # Use max steps as an arbitrary upper limit
        action = env.action_space.sample()  # Randomly select an action from the action space

        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        # Display the current step details
        #print(
        #    f"Step {step_count} - Action: {action}, Observation: {observation}, Reward: {reward}, Terminated: {terminated}")

    print(f"Episode {episode + 1} ended after {step_count} steps with total reward: {total_reward}\n")
    list_of_steps.append(step_count)

# Close the environment after use
env.close()
print("Average number of steps: ", statistics.mean(list_of_steps))