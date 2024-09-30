import numpy as np
import random
import gym
from gym import spaces

class MDPAlirezaStoch(gym.Env):
    # To Do: figure out how to implement episode logic (has to do with reset)
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=61):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the envs's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "envs": spaces.Box(0, size - 1, shape=(1,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(1,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(3)
        self.progress_states = [0, 1, 2, 3, 4, 5]
        self.trap_states = [6, 7]
        self.goal_states = [8, 9, 10]
        self.stochastic_states = list(range(11, 61))
        self.inv_goal_prob = 0.02
        self.probabilities = [1 - self.inv_goal_prob, self.inv_goal_prob / 2, self.inv_goal_prob / 2]

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        # create dictionary of action spaces. One for each state.
        # potential problem: two actions here have the same move. This is different in trap states, where each
        # action corresponds to unique move
        available_actions_progress = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0])]


        self._action_to_direction = {
            'state_0': {0: None,
                        1: None,
                        2: None},
            'state_1': {0: None,
                        1: None,
                        2: None},
            'state_2': {0: None,
                        1: None,
                        2: None},
            'state_3': {0: None,
                        1: None,
                        2: None},
            'state_4': {0: None,
                        1: None,
                        2: None},
            'state_5': {0: None,
                        1: None,
                        2: None}
        }

        # have to implement some shuffle logic - not every state has same action transition mapping
        # one action makes agent stay, one makes agent progress and one makes agent go to trap state
        # To Do: ensure that this happens each episode
        for state_key in self._action_to_direction:
            np.random.shuffle(available_actions_progress)
            counter = 0
            state_dict = self._action_to_direction[state_key]
            for d in state_dict:
                if state_dict[d] is None:
                    state_dict[d] = available_actions_progress[counter]
                    counter += 1


        # also have to implement shuffle logic
        # two actions make agent stay in trap states (either same or different trap state), one action makes agent go to state 1
        available_actions_trap = [np.array([0, 0]), np.array([1, 0]), np.array([0, 1])]

        self._action_to_direction_trap = {
            'state_6': {0: None,
                        1: None,
                        2: None},
            'state_7': {0: None,
                        1: None,
                        2: None}
        }

        for state_key in self._action_to_direction_trap:
            np.random.shuffle(available_actions_trap)
            counter = 0
            state_dict = self._action_to_direction_trap[state_key]
            for d in state_dict:
                if state_dict[d] is None:
                    state_dict[d] = available_actions_trap[counter]
                    counter += 1

        self._action_to_direction_stochastic = {
            f'state_{i}': {0: None, 1: None, 2: None} for i in range(11, 61)
        }

        available_actions_stoch = [np.array([0, 0]), np.array([1, 0]), np.array([0, 1])]

        for state_key in self._action_to_direction_stochastic:
            np.random.shuffle(available_actions_stoch)
            counter = 0
            state_dict = self._action_to_direction_stochastic[state_key]
            for d in state_dict:
                if state_dict[d] is None:
                    state_dict[d] = available_actions_stoch[counter]
                    counter += 1

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"envs": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):

        # randomly choose whether start location is state 1 or 2
        self._agent_location = random.choice([np.array([1, 0]), np.array([1,1])])
        # counts how often goal state was visited
        self.goal_counter = 0


        # target location
        self._target_location = [np.array([1, 8]), np.array([1, 9]), np.array([1, 10])]
        np.random.shuffle(self.progress_states)
        np.random.shuffle(self.trap_states)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info


    def step(self, action):
        # the following first checks whether the envs is currently in a progressing state or trap state
        # it then checks the action and depending on which action is taken from where it determines the next state
        # IMPORTANT: is self._agent_location up to date? Is it the state that the envs is in before taking this action?
        agent_absolute = False
        state_key = f"state_{self._agent_location[1]}"
        progress = True
        if self._agent_location[1] in self.progress_states:

            if np.array_equal(self._action_to_direction[state_key][action], np.array([1, 0])):
                # bad action - transported to trap state
                # To Do: use direction variable; check boundaries
                agent_absolute = True
                # randomize which trap state an envs is transported to from which progressing state upon taking bad action
                if self._agent_location[1] in (self.progress_states[0], self.progress_states[1], self.progress_states[2]):
                    self._agent_location = np.array([1, self.trap_states[0]])
                elif self._agent_location[1] in (self.progress_states[2], self.progress_states[3], self.progress_states[4]):
                    self._agent_location = np.array([1, self.trap_states[1]])
            elif np.array_equal(self._action_to_direction[state_key][action], np.array([0, 0])):
                agent_absolute = True
                if self._agent_location[1] == 3:
                    self._agent_location[1] = random.choice(self.stochastic_states)
                else:
                    self._agent_location = self._agent_location
            elif np.array_equal(self._action_to_direction[state_key][action], np.array([0, 1])):
                if self._agent_location[1] == 5:
                    agent_absolute = True
                    self._agent_location[1] = np.random.choice(self.goal_states, p = self.probabilities)
                    #self.goal_counter += 1
                    self._agent_location[1] = np.random.choice([0, 1])


        elif self._agent_location[1] in self.trap_states:
            progress = False
            # envs is in a trap state
            if np.array_equal(self._action_to_direction_trap[state_key][action], np.array([0, 0])):
                agent_absolute = True
                self._agent_location = self._agent_location
            elif np.array_equal(self._action_to_direction_trap[state_key][action], np.array([1, 0])):
                # good action - transported to state 1
                agent_absolute = True
                self._agent_location = np.array([1, 0])
            elif np.array_equal(self._action_to_direction_trap[state_key][action], np.array([0, 1])):
                agent_absolute = True
                if self._agent_location[1] == 6:
                    self._agent_location[1] = 7
                else:
                    self._agent_location[1] = 6


        elif self._agent_location[1] in self.stochastic_states:
            if np.array_equal(self._action_to_direction_stochastic[state_key][action], np.array([0, 0])):
                agent_absolute = True
                self._agent_location[1] = random.choice(self.stochastic_states)
            elif np.array_equal(self._action_to_direction_stochastic[state_key][action], np.array([1, 0])):
                agent_absolute = True
                self._agent_location[1] = random.choice(self.stochastic_states)
            elif np.array_equal(self._action_to_direction_stochastic[state_key][action], np.array([0, 1])):
                agent_absolute = True
                self._agent_location[1] = 3




            # Map the action (element of {0,1,2,3}) to the direction we walk in

        if not agent_absolute:
            # In theory, I do not have to make sure that envs doesn't leave environment, because this is taken
            # care of in the steps. It never has an action available with which it can leave environment

            direction = self._action_to_direction[state_key][action] if progress else self._action_to_direction_trap[state_key][action]
            self._agent_location = self._agent_location + direction



        # An episode is done iff the envs has reached the target
        terminated = True if self.goal_counter == 5 else False
        # think about the implementation of reward here - same name as reward in the agent, but different concept
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


