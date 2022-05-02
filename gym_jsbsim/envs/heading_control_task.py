from gym_jsbsim.task import Task
from gym_jsbsim.catalogs.catalog import Catalog as c
import math
import random
import numpy as np
import my_globals

"""
    A task in which the agent must perform steady, level flight following a certain heading.
    Every 150 sec a new target heading is set.
"""


class HeadingControlTask(Task):

    state_var = [
        c.attitude_pitch_rad, #pitch
        c.attitude_roll_rad, #roll
        c.attitude_psi_rad, #yaw
        c.velocities_v_down_fps, #velocity downwards [ft/s]
        # c.velocities_vc_fps, #"airspeed in knots"
        c.velocities_p_rad_sec, #roll rate
        c.velocities_q_rad_sec, #pitch rate
        c.velocities_r_rad_sec, #yaw rate
        c.velocities_u_fps, #"body frame x-axis velocity [ft/s]", should be 800
    ]

    action_var = [
        c.fcs_aileron_cmd_norm,
        c.fcs_elevator_cmd_norm,
        c.fcs_rudder_cmd_norm,
        c.fcs_throttle_cmd_norm,
    ]

    init_conditions = {
        c.ic_h_sl_ft: 10000,
        c.ic_terrain_elevation_ft: 0,
        c.ic_long_gc_deg: 1.442031,
        c.ic_lat_geod_deg: 43.607181,
        c.ic_u_fps: 800,
        c.ic_v_fps: 0,
        c.ic_w_fps: 0,
        c.ic_p_rad_sec: 0,
        c.ic_q_rad_sec: 0,
        c.ic_r_rad_sec: 0,
        c.ic_roc_fpm: 0,
        c.ic_psi_true_deg: 100,
        c.target_heading_deg: 100,
        c.target_altitude_ft: 10000,
        c.fcs_throttle_cmd_norm: 0.8,
        c.fcs_mixture_cmd_norm: 1,
        c.gear_gear_pos_norm: 0,
        c.gear_gear_cmd_norm: 0,
        c.steady_flight: 150,
    }

    def get_reward(self, state, sim):
        """
        Compute reward for task
        """
        heading_r = (my_globals.target_delta_heading - (sim.get_property_value(c.attitude_psi_deg) - my_globals.prev_heading)) ** 2
        alt_r = (my_globals.target_delta_alt - (sim.get_property_value(c.position_h_sl_ft) - my_globals.prev_alt)) ** 2
        deltar_r = (my_globals.target_delta_r - (sim.get_property_value(c.position_distance_from_start_mag_mt) - my_globals.prev_r)) ** 2
        
        # action_r = sim.get_property_value(c.fcs_aileron_cmd_norm) ** 2 + sim.get_property_value(c.fcs_elevator_cmd_norm) ** 2\
        #      + sim.get_property_value(c.fcs_rudder_cmd_norm) ** 2 + sim.get_property_value(c.fcs_throttle_cmd_norm) ** 2

        xspeed_r = (sim.get_property_value(c.velocities_u_fps) - 800) ** 2
        vdown_r = sim.get_property_value(c.velocities_v_down_fps) ** 2
        rpy_vel_r = sim.get_property_value(c.velocities_p_rad_sec) ** 2 +\
            sim.get_property_value(c.velocities_q_rad_sec) ** 2 +\
                sim.get_property_value(c.velocities_r_rad_sec) ** 2
        
        rpy_r = sim.get_property_value(c.attitude_roll_rad) ** 2\
            + sim.get_property_value(c.attitude_pitch_rad) ** 2\
                + sim.get_property_value(c.attitude_psi_rad) ** 2

        accel_r = sim.get_property_value(c.accelerations_n_pilot_x_norm) ** 2\
                    + sim.get_property_value(c.accelerations_n_pilot_y_norm) ** 2\
                    + (sim.get_property_value(c.accelerations_n_pilot_z_norm) + 1) ** 2

        # reward = -1 * (heading_r + alt_r + deltar_r + action_r + accel_r + roll_r + xspeed_r)
        reward = -1 * (heading_r + alt_r + deltar_r + rpy_vel_r + accel_r + rpy_r + xspeed_r + vdown_r)
        return reward

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

    def is_terminal(self, state, sim):
        # Change heading every 150 seconds
        my_globals.target_delta_r, my_globals.target_delta_heading, my_globals.target_delta_alt =\
            next_trajectory_point(delta_heading=sim.get_property_value(c.delta_heading), \
                delta_alt=sim.get_property_value(c.delta_altitude))
        my_globals.prev_r = sim.get_property_value(c.position_distance_from_start_mag_mt)
        my_globals.prev_heading = sim.get_property_value(c.attitude_psi_deg)
        my_globals.prev_alt = sim.get_property_value(c.position_h_sl_ft)

        if sim.get_property_value(c.simulation_sim_time_sec) >= sim.get_property_value(c.steady_flight):
            # If the target heading and altitude were not reached, we stop the simulation
            if math.fabs(sim.get_property_value(c.delta_heading)) > 10:
                return True
            if math.fabs(sim.get_property_value(c.delta_altitude)) >= 100:
                return True

            angle = int(sim.get_property_value(c.steady_flight) / 150) * 10
            sign = random.choice([+1.0, -1.0])
            new_heading = sim.get_property_value(c.target_heading_deg) + sign * angle
            new_heading = (new_heading + 360) % 360

            # print(f'Time to change: {sim.get_property_value(c.simulation_sim_time_sec)} (Heading: {sim.get_property_value(c.target_heading_deg)} -> {new_heading})')

            sim.set_property_value(c.target_heading_deg, new_heading)

            sim.set_property_value(c.steady_flight, sim.get_property_value(c.steady_flight) + 150)

        # if acceleration are too high stop the simulation
        acceleration_limit_x = 2.0  # "g"s
        acceleration_limit_y = 2.0  # "g"s
        acceleration_limit_z = 2.0  # "g"s
        if sim.get_property_value(c.simulation_sim_time_sec) > 10:
            if (
                math.fabs(sim.get_property_value(c.accelerations_n_pilot_x_norm)) > acceleration_limit_x
                or math.fabs(sim.get_property_value(c.accelerations_n_pilot_y_norm)) > acceleration_limit_y
                or math.fabs(sim.get_property_value(c.accelerations_n_pilot_z_norm) + 1) > acceleration_limit_z
            ):  # z component is expected to be -1 g
                return True

        # End up the simulation if the aircraft is on an extreme state
        # TODO: Is an altitude check needed?
        return (sim.get_property_value(c.position_h_sl_ft) < 3000) or bool(
            sim.get_property_value(c.detect_extreme_state)
        )
