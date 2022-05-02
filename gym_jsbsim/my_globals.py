def init():
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

def ret_list():
	return [target_delta_r, target_delta_heading, target_delta_alt]