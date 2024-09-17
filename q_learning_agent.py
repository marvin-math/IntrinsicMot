# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve as conv

class QLearningAgent:
  def __init__(self, env, params):
    self.n_actions = 4
    self.trans_prior = params['trans_prior']
    self.alpha = np.ones((env.size, self.n_actions, env.size)) * self.trans_prior
    self.p_N = np.ones(env.size) * (1 / env.size)
    self.pure_novelty = params['pure_novelty']
    self.pure_surprise = params['pure_surprise']

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

  def learn_environment(self, env, model_updater, planner, learning_rule, params, max_steps, n_episodes):
    # still to implement: surprise modulated updating of alpha
    # Start with a uniform value function
    value = np.ones((env.size, self.n_actions))
    #print(f"Value as initialised: {value}")

    # Run learning
    reward_sums = np.zeros(n_episodes)
    steps = np.zeros(n_episodes)
    model = np.nan*np.zeros((env.size, self.n_actions, 2)) # reward and next state
    #print(f"Model: {model}")

    # Loop over episodes
    for episode in range(n_episodes):
      observation, info = env.reset() # observation is agent location and target location
      terminated = False
      reward_sum = 0
      step_count = 0
      self.novelty_count = np.zeros(env.size)

      while not terminated and step_count < max_steps:
        # state before action
        state = observation["envs"][1]

        # should we increment this before the first action or after the first action?

        # choose next action
        action = self.epsilon_greedy(value[state], params['epsilon'])


        # observe outcome of action on environment
        observation, env_reward, terminated, truncated, info = env.step(action)


        #print(f"Reward after step: {reward}")
        step_count += 1

        # state after action
        next_state = observation["envs"][1]

        # compute novelty and update novelty count
        novelty = self.compute_novelty(env, self.novelty_count, step_count, next_state)
        self.novelty_count[next_state] += 1

        # compute surprise and update surprise count
        surprise = self.compute_surprise(self.alpha, state, action, next_state)
        self.alpha[state, action, next_state] += 1


        # update value function
        if self.pure_novelty:
          reward = novelty
        elif self.pure_surprise:
          reward = surprise
        value = learning_rule(state, action, reward, next_state, value, params)
        #print(f"Value: {value}")

        # update Dirichlet counts (pseudo counts for state transitions)
        # To Do: implement surprise modulation in these updates
        #print(f"Alpha: {self.alpha}")

        # update model
        #I think I have a model too much
        model = model_updater(model, state, action, reward, next_state)
        #print(f"Model after update: {model}")

        # planning
        value = planner(self.alpha, model, value, params)
        #print(f"Value after planning: {value}")

        # sum rewards obtained
        reward_sum += reward


      reward_sums[episode] = reward_sum
      steps[episode] = step_count
      print(model)
      print(self.novelty_count)


    return value, reward_sums, steps

  def compute_novelty(self, env, novelty_count, step_count, state):

    self.p_N[state] = (novelty_count[state] + 1) / (step_count + env.size)
    novelty_t = -np.log(self.p_N[state])
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


  def dyna_q_model_update(self, model, state, action, reward, next_state):
    """ Dyna-Q model update

    Args:
      model (ndarray): An array of shape (n_states, n_actions, 2) that represents
                       the model of the world i.e. what reward and next state do
                       we expect from taking an action in a state.
      state (int): the current state identifier
      action (int): the action taken
      reward (float): the reward received
      next_state (int): the transitioned to state identifier

    Returns:
      ndarray: the updated model
    """
    # Update our model with the observed reward and next state
    model[state, action] = reward, next_state

    return model


  def dyna_q_planning(self, alpha, model, value, params):
    """ Dyna-Q planning

    Args:
      model (ndarray): An array of shape (n_states, n_actions, 2) that represents
                       the model of the world i.e. what reward and next state do
                       we expect from taking an action in a state.
      value (ndarray): current value function of shape (n_states, n_actions)
      params (dict): a dictionary containing learning parameters

    Returns:
      ndarray: the updated value function of shape (n_states, n_actions)
    """
    # Perform k additional updates at random (planning)
    for _ in range(params['k']):
      # Find state-action combinations for which we've experienced a reward i.e.
      # the reward value is not NaN. The outcome of this expression is an Nx2
      # matrix, where each row is a state and action value, respectively.
      candidates = np.array(np.where(~np.isnan(model[:,:,0]))).T
      # Write an expression for selecting a random row index from our candidates
      idx = np.random.choice(len(candidates))

      # Obtain the randomly selected state and action values from the candidates
      state, action = candidates[idx]

      # Sample next state probabilities using the Dirichlet distribution
      transition_probs = np.random.dirichlet(alpha[state, action])
      next_state = np.random.choice(np.arange(alpha.shape[2]), p=transition_probs)
      surprise = -np.log(transition_probs[next_state])


      # Obtain the expected reward and next state from the model
      reward = model[state, action, 0]

      # Update the value function using Q-learning
      value = self.q_learning(state, action, reward, next_state, value, params)

    return value