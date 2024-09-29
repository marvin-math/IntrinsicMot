# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve as conv
import heapq
import copy

class QLearningAgent:
  def __init__(self, env, params):
    self.n_actions = 3 if params['stoch'] else 4
    self.trans_prior = params['trans_prior']
    self.alpha = np.ones((env.size, self.n_actions, env.size)) * self.trans_prior
    self.p_N = np.ones(env.size) * (1 / env.size)
    self.pure_novelty = params['pure_novelty']
    self.pure_surprise = params['pure_surprise']
    self.temperature = params['temperature']
    self.env_size = env.size

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

  def softmax(self,q):
    # TODO add temperature parameter
    """Compute softmax values for each sets of scores in x."""
    probabilities = np.exp(q/self.temperature) / np.sum(np.exp(q/self.temperature), axis=0)
    action = np.random.choice(len(q), p=probabilities)
    return action

  def learn_environment(self, env, r_f_updater, planner, learning_rule, params, max_steps, n_episodes):
    # still to implement: surprise modulated updating of alpha
    # Start with a uniform value function
    value = np.ones((env.size, self.n_actions))
    #print(f"Value as initialised: {value}")

    # Run learning
    reward_sums = np.zeros(n_episodes)
    steps = np.zeros(n_episodes)
    reward_fun = np.nan*np.zeros((env.size, self.n_actions)) # reward
    #print(f"Reward Function: {reward_fun}")

    # Loop over episodes
    for episode in range(n_episodes):
      observation, info = env.reset() # observation is agent location and target location
      terminated = False
      reward_sum = 0
      self.step_count = 0
      self.novelty_count = np.zeros(self.env_size)
      self.step_count_dict = {
                "10": None,
                "20": None,
                "30": None,
                "40": None,
                "50": None,
                "100": None,
                "200": None,
                "500": None,
                "1000": None,
                "2000": None,
                "5000": None,
                "10000": None}

      while not terminated and self.step_count < max_steps:
        # state before action
        state = observation["envs"][1]
        if self.step_count == 0:
          self.novelty_count[state] += 1

        # should we update q-value function before choosing the first action?
        action = self.softmax(value[state])


        # observe outcome of action on environment
        observation, env_reward, terminated, truncated, info = env.step(action)

        next_state = observation["envs"][1]


        # increase counts
        self.step_count += 1
        self.alpha[state, action, next_state] += 1
        self.novelty_count[next_state] += 1
        #update different novelty counters
        if self.step_count == 10:
          self.step_count_dict["10"] = copy.deepcopy(self.novelty_count)
          if self.pure_novelty:
            print(f"novelty at 10: {novelty}")
            print(f"novelty count at 10: {self.novelty_count}")
            print(f"value at 10: {value}")
        elif self.step_count == 20:
          self.step_count_dict["20"] = copy.deepcopy(self.novelty_count)
        elif self.step_count == 30:
          self.step_count_dict["30"] = copy.deepcopy(self.novelty_count)
          if self.pure_novelty:
            print(f"novelty at 30: {novelty}")
            print(f"novelty count at 30: {self.novelty_count}")
            print(f"value at 30: {value}")
        elif self.step_count == 40:
          self.step_count_dict["40"] = copy.deepcopy(self.novelty_count)
        elif self.step_count == 50:
          self.step_count_dict["50"] = copy.deepcopy(self.novelty_count)
          if self.pure_novelty:
            print(f"novelty at 50: {novelty}")
            print(f"novelty count at 50: {self.novelty_count}")
            print(f"value at 50: {value}")
        elif self.step_count == 100:
          self.step_count_dict["100"] = copy.deepcopy(self.novelty_count)
        elif self.step_count == 200:
          self.step_count_dict["200"] = copy.deepcopy(self.novelty_count)
          if self.pure_novelty:
            print(f"novelty at 200: {novelty}")
            print(f"novelty count at 200: {self.novelty_count}")
            print(f"value at 200: {value}")
        elif self.step_count == 500:
          self.step_count_dict["500"] = copy.deepcopy(self.novelty_count)
        elif self.step_count == 1000:
          self.step_count_dict["1000"] = copy.deepcopy(self.novelty_count)
          if self.pure_novelty:
            print(f"novelty at 1000: {novelty}")
            print(f"novelty count at 1000: {self.novelty_count}")
            print(f"value at 1000: {value}")
        elif self.step_count == 2000:
          self.step_count_dict["2000"] = copy.deepcopy(self.novelty_count)
        elif self.step_count == 5000:
          self.step_count_dict["5000"] = copy.deepcopy(self.novelty_count)
          if self.pure_novelty:
            print(f"novelty at 5000: {novelty}")
            print(f"novelty count at 5000: {self.novelty_count}")
            print(f"value at 5000: {value}")
        elif self.step_count == 9999:
          self.step_count_dict["10000"] = copy.deepcopy(self.novelty_count)
          if self.pure_novelty:
            print(f"novelty at 10000: {novelty}")
            print(f"novelty count at 10000: {self.novelty_count}")
            print(f"value at 10000: {value}")


        # compute novelty and update novelty count
        novelty = self.compute_novelty(self.env_size, self.novelty_count, self.step_count)

        # compute surprise and update surprise count
        surprise = self.compute_surprise(self.alpha, state, action, next_state)

        if self.pure_novelty:
          reward = novelty[next_state]
        elif self.pure_surprise:
          reward = surprise
        else:
          reward = env_reward

        # update reward function
        reward_fun = r_f_updater(reward_fun, state, action, reward)
        #print(f"reward_fun after update: {reward_fun}")

        # planning
        value = planner(self.alpha, reward_fun, value, params)
        #print(f"Value after planning: {value}")

        # sum rewards obtained
        reward_sum += reward


      reward_sums[episode] = reward_sum
      steps[episode] = self.step_count
      #print(reward_fun)
      #print(self.novelty_count)


    return value, reward_sums, steps

  def compute_novelty(self, size, novelty_count, step_count):
    self.p_N = (novelty_count + 1) / (step_count + size)
    novelty_t = -np.log(self.p_N)
    return novelty_t

  def compute_surprise(self, alpha, state, action, next_state):
    transition_probs = np.random.dirichlet(alpha[state, action])
    surprise = -np.log(transition_probs[next_state])
    return surprise

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
    #print(f"Q: {q}")

    # write an expression for finding the maximum Q-value at the current state
    max_next_q = np.max(value[int(next_state)])
    #print(f"Max next Q: {max_next_q}")

    # write the expression to compute the TD error
    td_error = reward + params['gamma'] * max_next_q - q
    #print(f"TD error: {td_error}")
    #print(f"Reward: {reward}")
    # write the expression that updates the Q-value for the state-action pair
    value[state, action] = q + params['alpha'] * td_error
    #print(f"Value after learning: {value}")

    return value


  def dyna_q_r_f_update(self, reward_fun, state, action, reward):
    """ Dyna-Q reward function update

    Args:
      reward_fun (ndarray): An array of shape (n_states, n_actions, 2) that represents
                       the reward_fun of the world i.e. what reward and next state do
                       we expect from taking an action in a state.
      state (int): the current state identifier
      action (int): the action taken
      reward (float): the reward received
      next_state (int): the transitioned to state identifier

    Returns:
      ndarray: the updated model
    """
    # Update our model with the observed reward and next state
    reward_fun[state, action] = reward
    #print(f"Reward Function after update: {reward_fun}")

    return reward_fun


  def prioritized_planning(self, alpha, reward_fun, value, params):
    """ Dyna-Q planning

    Args:
      reward_fun (ndarray): An array of shape (n_states, n_actions, 2) that represents
                       the model of the world i.e. what reward and next state do
                       we expect from taking an action in a state.
      value (ndarray): current value function of shape (n_states, n_actions)
      params (dict): a dictionary containing learning parameters

    Returns:
      ndarray: the updated value function of shape (n_states, n_actions)
    """

    # Initialize a priority queue
    priority_queue = []

    # find all state, action pairs that have been visited
    candidates = np.array(np.where(~np.isnan(reward_fun[:, :]))).T

    for state, action in candidates:
      # Sample next state probabilities using the Dirichlet distribution
      transition_probs = np.random.dirichlet(alpha[state, action])
      next_state = np.random.choice(np.arange(alpha.shape[2]), p=transition_probs)

      reward = reward_fun[state, action]

      # Q-value of current state-action pair
      q = value[state, action]

      # finding the maximum Q-value at the current state
      max_next_q = np.max(value[int(next_state)])

      #compute the TD error
      td_error = reward + params['gamma'] * max_next_q - q

      # Add the state-action pair to the priority queue based on the TD error
      heapq.heappush(priority_queue, (-abs(td_error), (state, action)))

    for _ in range(min(params['k'], len(priority_queue))):
      # Pop the highest-priority state-action pair
      priority, (state, action) = heapq.heappop(priority_queue)

      # Sample next state probabilities using the Dirichlet distribution
      transition_probs = np.random.dirichlet(alpha[state, action])
      next_state = np.random.choice(np.arange(alpha.shape[2]), p=transition_probs)

      # Get the reward from the model
      reward = reward_fun[state, action]
      #print(f"Reward in planning: {reward}")

      # Update the Q-value using Q-learning
      value = self.q_learning(state, action, reward, next_state, value, params)

    return value