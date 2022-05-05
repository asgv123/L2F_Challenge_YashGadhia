target_delta_r = 0
target_delta_heading = 0
target_delta_alt = 0
prev_r = 0
prev_heading = 0
prev_alt = 0
is_debug = 0

class MyGlobals:
	def my_g_init():
		global target_delta_r
		global target_delta_heading
		global target_delta_alt
		global prev_r
		global prev_heading
		global prev_alt
		global is_debug
		target_delta_r = 0
		target_delta_heading = 0
		target_delta_alt = 0
		prev_r = 0
		prev_heading = 0
		prev_alt = 10000
		is_debug = False

	def my_g_ret_list():
		global target_delta_r
		global target_delta_heading
		global target_delta_alt
		return [target_delta_r, target_delta_heading, target_delta_alt]

	# def ret_name_list():
	# 	return ["target_delta_r", "target_delta_heading", "target_delta_alt"]

	def my_g_reset():
		global target_delta_r
		global target_delta_heading
		global target_delta_alt
		global prev_r
		global prev_heading
		global prev_alt
		global is_debug
		target_delta_r = 0
		target_delta_heading = 0
		target_delta_alt = 0
		prev_r = 0
		prev_heading = 0
		prev_alt = 10000
		is_debug = False

def next_trajectory_point(delta_heading, delta_alt, dt=150/1800, turning_angle=20, turning_time=20, turning_speed=800, forward_speed=800, eps=1):
	# given speed(fps), delta_heading(deg), turning_angle(deg), turning_time(s), can find next trajectories
	# take epsilon as threshold on delta_heading
	# return delta_r in m, delta_heading in deg, delta_alt in ft
	if abs(delta_heading) <= eps:
		return dt * forward_speed, 0, (delta_alt / turning_time) * dt
	else:
		delta_r = dt * turning_speed
		delta_theta = turning_angle / (turning_time / dt)
		if delta_heading < 0:
			delta_theta *= -1
		return delta_r, delta_theta, (delta_alt / turning_time) * dt
	pass