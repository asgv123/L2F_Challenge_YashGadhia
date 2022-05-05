from types import MethodType
import numpy as np
import gym
from gym.spaces import Box, Discrete
from gym_jsbsim.catalogs.catalog import Catalog
from gym_jsbsim import my_globals

class Task:
    """
    A class to subclass in order to create a task with its own observation variables,
    action variables, termination conditions and agent_reward function.
    """

    action_var = None
    state_var = None
    init_conditions = None
    output = state_var
    jsbsim_freq = 60
    agent_interaction_steps = 5
    aircraft_name = "A320"

    def __init__(self):

        # set default output to state_var
        if self.output is None:
            self.output = self.state_var

        # modify Catalog to have only the current task properties
        names_away = []
        for name, prop in Catalog.items():
            if not (
                prop in self.action_var
                or prop in self.state_var
                or prop in self.init_conditions
                or prop in self.output
            ):
                names_away.append(name)
        for name in names_away:
            Catalog.pop(name)

    def get_reward(self, state, sim):
        return 0

    def is_terminal(self, state, sim):
        return False

    def get_observation_var(self):
        return self.state_var

    def get_action_var(self):
        return self.action_var

    def get_initial_conditions(self):
        return self.init_conditions

    def get_output(self):
        return self.output

    def get_observation_space(self):
        """
        Get the task's observation Space object
        :return : spaces.Tuple composed by spaces of each property.
        """

        low = np.empty(shape=(0,))
        high = np.empty(shape=(0,))

        for prop in self.state_var:
            if prop.spaces is Box:
                low = np.concatenate([low,np.array([prop.min])])
                high = np.concatenate([high,np.array([prop.max])])
            elif prop.spaces is Discrete:
                raise TypeError("Discrete state space found")
        
        #target_delta_r, target_delta_heading (deg), target_delta_alt
        np.concatenate([low, np.array([-67, -1, -50])])
        np.concatenate([high, np.array([67, 1, 50])])
        
        return Box(low=low, high=high, dtype="float")

    def get_action_space(self):
        """
        Get the task's action Space object
        :return : spaces.Tuple composed by spaces of each property.
        """
        low = np.empty(shape=(0,))
        high = np.empty(shape=(0,))

        for prop in self.action_var:
            if prop.spaces is Box:
                low = np.concatenate([low,np.array([prop.min])])
                high = np.concatenate([high,np.array([prop.max])])
            elif prop.spaces is Discrete:
                raise TypeError("Discrete action space found")
        return Box(low=low, high=high, dtype="float")

    def render(self, sim, mode="human", **kwargs):
        pass

    def define_aircraft(self, aircraft_name="A320"):
        self.aircraft_name = aircraft_name

    def define_state(self, states=None):
        self.state_var = states

    def define_action(self, actions=None):
        self.action_var = actions

    def define_init_conditions(self, init_conditions=None):
        self.init_conditions = init_conditions

    def define_output(self, output=None):
        self.output = output

    def define_jsbsim_freq(self, freq=60):
        self.jsbsim_freq = freq

    def define_agent_interaction_steps(self, steps=5):
        self.agent_interaction_steps = steps

    def define_reward(self, func):
        self.get_reward = MethodType(func, self)

    def define_is_terminal(self, func):
        self.is_terminal = MethodType(func, self)