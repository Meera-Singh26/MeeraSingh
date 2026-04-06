import os

print("Select Patient Problem:")
print("1. Knee Pain")
print("2. Neck Pain")
print("3. Muscle Stress")
print("4. Hip Displacement")

problem = input("Enter problem choice: ")

# pass to other scripts
os.environ["PATIENT_PROBLEM"] = problem

print("\n1. Train Model")
print("2. Test Model")

choice = input("Enter choice: ")

if choice == "1":
    os.system("python -m train.train_ppo")
elif choice == "2":
    os.system("python -m test.test_model")
else:
    print("Invalid choice")