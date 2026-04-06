from stable_baselines3 import PPO
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.rehab_camera_env import RehabCameraEnv

# read patient problem from main.py
problem = int(os.environ.get("PATIENT_PROBLEM", 1))

# exercise mapping
def get_exercise(problem):
    if problem == 1:
        return "knee"
    elif problem == 2:
        return "neck"
    elif problem == 3:
        return "muscle"
    elif problem == 4:
        return "hip"
    else:
        return "knee"

exercise = get_exercise(problem)

print("🚀 Starting test...")
print(f"🧠 Detected Problem Type: {exercise.upper()}")

# Create env
env = DummyVecEnv([lambda: RehabCameraEnv()])

# Load normalization
env = VecNormalize.load("models/vec_normalize.pkl", env)
env.training = False
env.norm_reward = False

# Load model
model = PPO.load("models/ppo_rehab")

obs = env.reset()

for step in range(500):

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, info = env.step(action)

    # feedback based on problem
    if exercise == "knee":
        if action == 0:
            print("⬆️ Raise knee")
        elif action == 1:
            print("⏸ Hold knee")
        else:
            print("⬇️ Lower knee slowly")

    elif exercise == "neck":
        if action == 0:
            print("⬅️ Tilt neck left")
        elif action == 1:
            print("⏸ Hold neck")
        else:
            print("➡️ Tilt neck right")

    elif exercise == "hip":
        if action == 0:
            print("⬆️ Lift hip")
        elif action == 1:
            print("⏸ Hold hip")
        else:
            print("⬇️ Lower hip slowly")

    elif exercise == "muscle":
        if action == 0:
            print("🧘 Stretch muscle")
        elif action == 1:
            print("⏸ Hold stretch")
        else:
            print("🔄 Relax muscle")

    if done[0]:
        obs = env.reset()

print("✅ Test completed")