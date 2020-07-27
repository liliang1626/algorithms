import numpy as np


# Number of total trials 
TRIAL_COUNT = 10000
# Number of arms in the trial
ARMS_COUNT = 10


# Initialize the real reward probability for each arm uniformly randomly
true_rewards_prob = np.random.uniform(low=0, high=1, size=ARMS_COUNT)
# Observed reward probability with trials so far
observed_rewards_prob = np.zeros(ARMS_COUNT)
# Number of times each arm gets selected
selected_count = np.zeros(ARMS_COUNT)
# Total rewards with trials so far
total_rewards = 0

def compute_delta(trial_idx, arm_idx):
    if selected_count[arm_idx] == 0:
        return 1
    else:
        # Based on Chernoff-Hoeffding Bound
        return np.sqrt(2 * np.log(trial_idx) / selected_count[arm_idx])

def upper_confidence_bound(trial_idx, arms_count):
    upper_bound_rewards_prob = [observed_rewards_prob[j] + compute_delta(trial_idx, j) for j in range(ARMS_COUNT)]
    arm_idx = np.argmax(upper_bound_rewards_prob)
    reward = np.random.binomial(n=1, p=true_rewards_prob[arm_idx])
    return arm_idx, reward

for i in range(TRIAL_COUNT):
    arm_idx, reward = upper_confidence_bound(i, ARMS_COUNT)

    # Existing total rewards for arms[arm_idx]
    arm_rewards = observed_rewards_prob[arm_idx] * selected_count[arm_idx]
    
    # Update the number of times arm[arm_idx] gets selected 
    selected_count[arm_idx] += 1

    # Update the observed reward probability
    observed_rewards_prob[arm_idx] = (arm_rewards + reward) / selected_count[arm_idx]

    # Update total rewards
    total_rewards += reward

    print(f"Trial {i}/{TRIAL_COUNT} - selected arm {arm_idx} with reward {reward}")

print(f"true_rewards_prob = {true_rewards_prob}")
print(f"total_rewards = {total_rewards}")