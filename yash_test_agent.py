import gym
import gym_jsbsim
import math
import numpy as np
import matplotlib.pyplot as plt

#action limits
ROLL_U = 1.0
ROLL_L = -1.0
PITCH_U = 1.0
PITCH_L = -1.0
YAW_U = 1.0
YAW_L = -1.0
SPEED_U = 0.9
SPEED_L = 0.0

def clip_actions(action):
	new_action_tuple = ()
	new_action_tuple += (np.array(max(ROLL_L, min(action[0], ROLL_U))),)
	new_action_tuple += (np.array(max(PITCH_L, min(action[1], PITCH_U))),)
	new_action_tuple += (np.array(max(YAW_L, min(action[2], YAW_U))),)
	new_action_tuple += (np.array(max(SPEED_L, min(action[3], SPEED_U))),)
	# print(new_action_tuple)
	return new_action_tuple

env = gym.make("GymJsbsim-HeadingControlTask-v0")

x = 0
y = 0
step = 1
plt.axis([-100, 100, -100, 100])
plt.plot(np.array([0, 100]),np.array([0, 0]), color='green')
#plt.plot(np.array([0, 100*math.cos((state[0]/180)*math.pi)]),np.array([0, 100*math.sin((state[0]/180)*math.pi)]), color='red')
plt.scatter(x, y, s=5, c='black')
plt.pause(0.1)

dt = 1/60.0

curr_state = env.reset()
done = False

kp = 0.05
kd = 0
ki = 0
previous_error = 0
integral = 0 

steps = 0
tot_rew = 0
delta_head_hist = []

while not done:
	#heading PID
	error = curr_state[1]
	delta_head_hist.append(error)
	integral = min(integral + error * dt, 10)
	derivative = (error - previous_error) / dt
	output = kp * error + ki * integral + kd * derivative
	previous_error = error
	my_action = (np.array(0), np.array(0), np.array(output), np.array(0))
	clipped_action = clip_actions(my_action)
	curr_state, reward, done, _ = env.step(clipped_action)

	steps += 1
	tot_rew += reward

	# if steps % 1000 == 0:
	# 	print("Error: ", error, "\n")
	# 	print("Action: ", clipped_action, '\n')
	# 	plt.plot(delta_head_hist)
	
	# print("action =", clipped_action, " ---> State =", state, " : Reward =", reward)

	x = x + step*math.cos((curr_state[1]/180)*math.pi)
	y = y + step*math.sin((curr_state[1]/180)*math.pi)
	#plt.plot(np.array([0, 100]),np.array([0, 0]), color='green')
	#plt.plot(np.array([0, 100*math.cos((state[0]/180)*math.pi)]),np.array([0, 100*math.sin((state[0]/180)*math.pi)]), color='red')
	plt.scatter(x, y, s=5, c='black')
	#plt.plot(np.array([x, x+1000*math.cos((state[0]/180)*math.pi)]),np.array([y, y+1000*math.sin((state[0]/180)*math.pi)]), color='red')
	plt.pause(0.1)

print("#Steps: ", steps, "\nReward: ", tot_rew, "\n")
plt.show()
