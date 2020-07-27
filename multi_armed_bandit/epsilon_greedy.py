import numpy as np


# Number of total trials 
TRIAL_COUNT = 10000
# Number of arms in the trial
ARMS_COUNT = 10
# Probability of choosing one arm randomly ignoring currently observed best arm
EPSILON = 0.01


# Initialize the real reward probability for each arm uniformly randomly
true_rewards_prob = np.random.uniform(low=0, high=1, size=ARMS_COUNT)
# Observed reward probability with trials so far
observed_rewards_prob = np.zeros(ARMS_COUNT)
# Number of times each arm gets selected
selected_count = np.zeros(ARMS_COUNT)
# Total rewards with trials so far
total_rewards = 0

def epsilon_greedy(arms_count, epsilon):
    # Randomly choose arm at the probability of epsilon
    if np.random.random() < epsilon:
        idx = np.random.randint(low=0, high=ARMS_COUNT)
        is_random = True
    else:
        idx = np.argmax(observed_rewards_prob)
        is_random = False
    reward = np.random.binomial(n=1, p=true_rewards_prob[idx])
    return idx, reward, is_random

for i in range(TRIAL_COUNT):
    # Repeat the trial of picking a arm to maximize the total rewards
    arm_idx, reward, is_random = epsilon_greedy(ARMS_COUNT, EPSILON)

    # Existing total rewards for arms[arm_idx]
    arm_rewards = observed_rewards_prob[arm_idx] * selected_count[arm_idx]
    
    # Update the number of times arm[arm_idx] gets selected 
    selected_count[arm_idx] += 1

    # Update the observed reward probability
    observed_rewards_prob[arm_idx] = (arm_rewards + reward) / selected_count[arm_idx]

    # Update total rewards
    total_rewards += reward

    randomly = "randomly " if is_random else ""
    print(f"Trial {i}/{TRIAL_COUNT} - {randomly}selected arm {arm_idx} with reward {reward}")

print(f"true_rewards_prob = {true_rewards_prob}")
print(f"total_rewards = {total_rewards}")