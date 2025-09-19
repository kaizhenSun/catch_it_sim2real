from gymnasium.envs.registration import register

register(
    id="gym_dcmm/DcmmVecWorld-v1",
    entry_point="gym_dcmm.envs:DcmmVecEnvArm",
    #  max_episode_steps=300,
)