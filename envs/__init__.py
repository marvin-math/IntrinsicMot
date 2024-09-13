from gym.envs.registration import register
from agent.environment import MDPAlireza

register(
    id="MDPAlireza-v0",
    entry_point="agent.environment:MDPAlireza",
)