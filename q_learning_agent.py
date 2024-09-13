# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve as conv

class QLearningAgent:

  def epsilon_greedy(self, q, epsilon):
    """Epsilon-greedy policy: selects the maximum value action with probabilty
    (1-epsilon) and selects randomly with epsilon probability.

    Args:
      q (ndarray): an array of action values
      epsilon (float): probability of selecting an action randomly

    Returns:
      int: the chosen action
    """
    if np.random.random() > epsilon:
      action = np.argmax(q)
    else:
      action = np.random.choice(len(q))

    return action

  def learn_environment(self, env, learning_rule, params, max_steps, n_episodes):
    # Start with a uniform value function
    value = np.ones((env.size, 4))

    # Run learning
    reward_sums = np.zeros(n_episodes)
    steps = np.zeros(n_episodes)

    # Loop over episodes
    for episode in range(n_episodes):
      observation, info = env.reset() # observation is agent location and target location
      terminated = False
      reward_sum = 0
      step_count = 0

      while not terminated and step_count < max_steps:
        # state before action
        state = observation["envs"][1]
        # choose next action
        action = self.epsilon_greedy(value[state], params['epsilon'])


        # observe outcome of action on environment
        observation, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        # state after action
        next_state = observation["envs"][1]

        # update value function
        value = learning_rule(state, action, reward, next_state, value, params)

        # sum rewards obtained
        reward_sum += reward


      reward_sums[episode] = reward_sum
      steps[episode] = step_count

    return value, reward_sums, steps

  def q_learning(self, state, action, reward, next_state, value, params):
    """Q-learning: updates the value function and returns it.

    Args:
      state (int): the current state identifier
      action (int): the action taken
      reward (float): the reward received
      next_state (int): the transitioned to state identifier
      value (ndarray): current value function of shape (n_states, n_actions)
      params (dict): a dictionary containing the default parameters

    Returns:
      ndarray: the updated value function of shape (n_states, n_actions)
    """
    # Q-value of current state-action pair
    q = value[state, action]

    # write an expression for finding the maximum Q-value at the current state
    max_next_q = np.max(value[next_state])

    # write the expression to compute the TD error
    td_error = reward + params['gamma'] * max_next_q - q
    # write the expression that updates the Q-value for the state-action pair
    value[state, action] = q + params['alpha'] * td_error

    return value


