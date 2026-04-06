from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.mujoco_env import RehabHumanoidEnv
from env.rehab_camera_env import RehabCameraEnv

# Create env
env = DummyVecEnv([lambda: RehabHumanoidEnv()])

env = DummyVecEnv([lambda: RehabCameraEnv()])

# Normalize
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# PPO model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1
)

# Train
model.learn(total_timesteps=5000)

# Save
model.save("models/ppo_rehab")
env.save("models/vec_normalize.pkl")