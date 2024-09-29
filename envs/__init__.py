from gym.envs.registration import register

register(
    id="MDPAlireza-v0",
    entry_point="envs.environment:MDPAlireza",
)

register(
    id = 'MDPAlireza-vs',
    entry_point="envs.stoch_environment:MDPAlirezaStoch",
)