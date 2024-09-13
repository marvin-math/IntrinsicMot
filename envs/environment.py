import numpy as np

import gym
from gym import spaces

class MDPAlireza(gym.Env):
    # To Do: figure out how to implement episode logic (has to do with reset)
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=11):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(1,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(1,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        # create dictionary of action spaces. One for each state.
        # potential problem: two actions here have the same move. This is different in trap states, where each
        # action corresponds to unique move
        available_actions_progress = [np.array([0, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([-1, 0])]


        self._action_to_direction = {
            'state_0': {0: None, #neutral
                        1: None, #good
                        2: None, #up
                        3: None}, #up
            'state_1': {0: None, #neutral
                        1: None, #good
                        2: None, #up
                        3: None},  # left
            'state_2': {0: None, #neutral
                        1: None, #good
                        2: None, #up
                        3: None},  # left
            'state_3': {0: None, #neutral
                        1: None, #good
                        2: None, #up
                        3: None},  # left
            'state_4': {0: None, #neutral
                        1: None, #good
                        2: None, #up
                        3: None},  # left
            'state_5': {0: None, #neutral
                        1: None, #good
                        2: None, #up
                        3: None},  # left
            'state_6': {0: None,  # neutral
                        1: None,  # good
                        2: None,  # up
                        3: None}
        }

        # have to implement some shuffle logic - not every state has same action transition mapping
        # two bad actions, one good action, one neutral
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
        # three actions make agent stay or go to other trap state, one action to state 1
        available_actions_trap = [np.array([0, 0]), np.array([0, -1]), np.array([-1, 0]), np.array([0, 1])]

        self._action_to_direction_trap = {
            'state_7': {0: None, #neutral
                        1: None, #good
                        2: None, #up
                        3: None}, #left
            'state_8': {0: None, #neutral
                        1: None, #good
                        2: None, #up
                        3: None},  # left
            'state_9': {0: None, #neutral
                        1: None, #good
                        2: None, #up
                        3: None},  # left
        }

        for state_key in self._action_to_direction_trap:
            np.random.shuffle(available_actions_trap)
            counter = 0
            state_dict = self._action_to_direction_trap[state_key]
            for d in state_dict:
                if state_dict[d] is None:
                    state_dict[d] = available_actions_trap[counter]
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
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):

        # Choose the agent's location
        self._agent_location = np.array([1, 0])

        # target location
        self._target_location = np.array([1, 10])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info


    def step(self, action):
        # the following first checks whether the agent is currently in a progressing state or trap state
        # it then checks the action and depending on which action is taken from where it determines the next state
        # IMPORTANT: is self._agent_location up to date? Is it the state that the agent is in before taking this action?
        agent_absolute = False
        progress_states = [0, 1, 2, 3, 4, 5, 6]
        trap_states = [7, 8, 9]
        state_key = f"state_{self._agent_location[1]}"
        progress = True
        if self._agent_location[1] in progress_states:
            np.random.shuffle(progress_states)
            np.random.shuffle(trap_states)
            if np.equal(self._action_to_direction[state_key][action], np.array([-1, 0])):
                # bad action - transported to trap state
                # To Do: use direction variable; check boundaries
                agent_absolute = True
                # randomize which trap state an agent is transported to from which progressing state upon taking bad action
                if self._agent_location[1] in (progress_states[0], progress_states[1]):
                    self._agent_location = np.array([1, trap_states[0]])
                elif self._agent_location[1] in (progress_states[2], progress_states[3]):
                    self._agent_location = np.array([1, trap_states[1]])
                elif self._agent_location[1] in (progress_states[4], progress_states[5], progress_states[6]):
                    self._agent_location = np.array([1, trap_states[2]])
            elif np.equal(self._action_to_direction[state_key][action], np.array([0, 0])):
                agent_absolute = True
                self._agent_location = self._agent_location
            elif np.equal(self._action_to_direction[state_key][action], np.array([0, 1])):
                if self._agent_location[1] == 6:
                    agent_absolute = True
                    self._agent_location[1] = 10


        elif self._agent_location[1] in trap_states:
            progress = False
            # agent is in a trap state
            if np.equal(self._action_to_direction_trap[state_key][action], np.array([-1, 0])):
                #stay at position
                agent_absolute = True
                self._agent_location = self._agent_location
            elif np.equal(self._action_to_direction_trap[state_key][action], np.array([0, 0])):
                # good action - transported to state 1
                agent_absolute = True
                self._agent_location = np.array([1, 0])
            elif np.equal(self._action_to_direction_trap[state_key][action], np.array([0, -1])):
                # one to the left
                if self._agent_location[1] not in (8, 9):
                    agent_absolute = True
                    self._agent_location[1] = 9
            elif self._action_to_direction_trap[state_key][action] == np.array([0, 1]):
                # one to the right
                if self._agent_location[1] not in (7, 8):
                    agent_absolute = True
                    self._agent_location[1] = 7

            # Map the action (element of {0,1,2,3}) to the direction we walk in

        if not agent_absolute:
            # In theory, I do not have to make sure that agent doesn't leave environment, because this is taken
            # care of in the steps. It never has an action available with which it can leave environment

            direction = self._action_to_direction[state_key][action] if progress else self._action_to_direction_trap[state_key][action]
            self._agent_location = self._agent_location + direction



            # An episode is done iff the agent has reached the target
            terminated = np.array_equal(self._agent_location, self._target_location)
            reward = 1 if terminated else 0  # Binary sparse rewards
            observation = self._get_obs()
            info = self._get_info()

            if self.render_mode == "human":
                self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


