import numpy as np


# Number of total trials
TRIAL_COUNT = 10000
# Number of arms in the trial
ARMS_COUNT = 10


# Initialize the real reward probability for each arm uniformly randomly
true_rewards_prob = np.random.uniform(low=0, high=1, size=ARMS_COUNT)
# Alpha parameter in Beta distribution for each arm
alpha = np.ones(ARMS_COUNT)
# Beta parameter in Beta distribution for each arm
beta = np.ones(ARMS_COUNT)
# Total rewards with trials so far
total_rewards = 0

def thompson_sampling():
    arms_rewards_prob = [np.random.beta(alpha[j], beta[j]) for j in range(ARMS_COUNT)]
    arm_idx = np.argmax(arms_rewards_prob)
    reward = np.random.binomial(n=1, p=true_rewards_prob[arm_idx])
    return arm_idx, reward

for i in range(TRIAL_COUNT):
    arm_idx, reward = thompson_sampling()

    # Update the number of times arm[arm_idx] gets selected
    alpha[arm_idx] += reward
    beta[arm_idx] += (1 - reward)

    # Update total rewards
    total_rewards += reward

    print(f"Trial {i}/{TRIAL_COUNT} - selected arm {arm_idx} with reward {reward}")

print(f"true_rewards_prob = {true_rewards_prob}")
print(f"total_rewards = {total_rewards}")
