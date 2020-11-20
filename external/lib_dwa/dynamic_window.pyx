# cython: profile=False
# distutils: language=c++

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
cimport cython
from math import sqrt
from libc.math cimport cos as ccos
from libc.math cimport sin as csin
from libc.math cimport acos as cacos
from libc.math cimport sqrt as csqrt

import os
from yaml import load
from matplotlib.pyplot import imread




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def linear_dwa(np.float32_t[:] s_next,
        np.float32_t[:] angles,
        u, v, gx, gy, dt,
        DV=0.1, # velocity sampling resolution
        UMIN=-0.5,
        UMAX=0.5,
        VMIN=-0.5,
        VMAX=0.5,
        AMAX=0.4,
        NOISE=0., # adds randomness to the sampled velocity, in the hope to reach otherwise unreachable velocities
        COMFORT_RADIUS_M=0.5,
        PLANNING_RADIUS_M=0.7,
        ):

    # Specification vel limits
    W = [UMIN, UMAX, VMIN, VMAX]




    # sample in window
    cdef np.float32_t best_score = -1000
    cdef np.float32_t best_u = 0
    cdef np.float32_t best_v = 0
    cdef np.float32_t du = 0
    cdef np.float32_t dv = 0
    cdef np.float32_t norm_duv = 0
    cdef np.float32_t max_norm_duv = AMAX * dt
    cdef np.float32_t max_norm_uv = VMAX
    cdef np.float32_t us = 0
    cdef np.float32_t vs = 0
    cdef np.float32_t xs = 0
    cdef np.float32_t ys = 0
    cdef np.float32_t dgxs = 0 # sampled x dist to goal
    cdef np.float32_t dgys = 0 # sampled y dist to goal
    cdef np.float32_t min_dist = 100
    cdef np.float32_t min_dist_s = 100
    cdef np.float32_t improvement = 0
    cdef np.float32_t scan_score = 0
    cdef np.float32_t goal_score = 0
    cdef np.float32_t score = 0
    cdef np.float32_t cgx = np.float32(gx)
    cdef np.float32_t cgy = np.float32(gy)
    cdef np.float32_t goal_norm = csqrt(gx * gx + gy * gy)
    cdef np.float32_t goal_norm_s = 0
    cdef np.float32_t cMAX_GOAL_NORM = 3
    cdef np.float32_t cdt = np.float32(dt)
    cdef np.float32_t cCOMFORT_RADIUS_M = np.float32(COMFORT_RADIUS_M)
    cdef np.float32_t cPLANNING_RADIUS_M = np.float32(PLANNING_RADIUS_M)
    cdef np.float32_t[:] us_list = np.append(np.arange(UMIN, UMAX, DV), [0]).astype(np.float32)
    cdef np.float32_t[:] vs_list = np.append(np.arange(VMIN, VMAX, DV), [0]).astype(np.float32)
    cdef np.float32_t[:] noise_u = np.append(np.random.normal(0, NOISE, size=(len(us_list)-1)), [0]).astype(np.float32)
    cdef np.float32_t[:] noise_v = np.append(np.random.normal(0, NOISE, size=(len(vs_list)-1)), [0]).astype(np.float32)
    cdef np.float32_t[:] s_next_shift = np.zeros_like(s_next, dtype=np.float32)
    # find current closest point
    for k in range(len(s_next)):
        r = s_next[k] 
        if r == 0:
            continue
        if r < min_dist:
            min_dist = r
    # sample in window and score
    for i in range(len(us_list)):
        us = us_list[i] + noise_u[i]
        for j in range(len(vs_list)):
            vs = vs_list[j] + noise_v[j]
            # dynamic limits as condition (circle around current u v)
            du = us - u
            dv = vs - v
            norm_duv = csqrt(du * du + dv * dv)
            if norm_duv > max_norm_duv:
                continue
            # specification vel limits corner case (ellipsid around 0, 0)
            norm_uv = csqrt(us * us + vs * vs)
            if norm_uv > max_norm_uv:
                continue
            # motion model TODO refine
            xs = us * cdt
            ys = vs * cdt
            min_dist_s = 100
            # goal score is the diference between the current goal distance and the sampled one
            dgxs = cgx - xs
            dgys = cgy - ys
            goal_norm_s = csqrt(dgxs * dgxs + dgys * dgys)
            goal_score = goal_norm - goal_norm_s
            # refuse to exceed max goal dist
            if goal_norm > cMAX_GOAL_NORM:
                if goal_score < 0:
                    continue
            # scan score - shift the scan by dx, dy, then find the smallest range (closest point)
            for k in range(len(s_next)):
                # TODO potential optim. : precompute smaller scan with only ranges close enough to matter
                r = s_next[k] 
                if r == 0:
                    continue
                shifted_r = r - xs * ccos(angles[k]) - ys * csin(angles[k])
                s_next_shift[k] = shifted_r
                if shifted_r < min_dist_s:
                    min_dist_s = shifted_r
            if min_dist > cCOMFORT_RADIUS_M:
                # normal situation
                if min_dist_s < cCOMFORT_RADIUS_M:
                    scan_score = 0
#                 elif min_dist_s < cPLANNING_RADIUS_M: # linear ramp 0 to 1
#                     scan_score = (min_dist_s - cCOMFORT_RADIUS_M) / (cPLANNING_RADIUS_M - cCOMFORT_RADIUS_M)
                else:
                    scan_score = 1
            else:
                # we are far inside an obstacle, priority is evasion TODO: improve 
                improvement = min_dist_s - min_dist
                if improvement > 0:
                    scan_score = improvement + 0.1 * goal_score
                    goal_score = 1
                else:
                    scan_score = 0
            score = scan_score * goal_score
#             print("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(gx, gy, xs, ys, goal_score, score))
            if score > best_score:
                best_score = score
                best_u = us
                best_v = vs

    return best_u, best_v, best_score


# below for diff-drive DWA -----------------------------------------------------------------

def angle_difference_rad(target_angle, angle):
    """     / angle
           /
          / d
         /)___________ target
    """
    delta_angle = angle - target_angle
    delta_angle = np.arctan2(np.sin(delta_angle), np.cos(delta_angle))  # now in [-pi, pi]
    return delta_angle

class NavigationPlanner(object):
    robot_radius = 0.3  # [m]
    safety_distance = 0.1  # [m]
    robot_max_speed = 1.  # [m/s]
    robot_max_w = 1. # [rad/s]
    robot_max_accel = 0.5  # [m/s^2]
    robot_max_w_dot = 10.  # [rad/s^2]

    def __init__(self):
        raise NotImplementedError

    def set_static_obstacles(self, static_obstacles):
        self.static_obstacles = static_obstacles

    def compute_cmd_vel(self, crowd, robot_pose, goal, show_plot=True, debug=False):
        raise NotImplementedError

class DynamicWindowApproachNavigationPlanner(NavigationPlanner):
    def __init__(self):
        # dynamic window parameters
        self.speed_resolution = 0.01  # [m/s]
        self.rot_resolution = 0.02  # [rad/s]
        self.prediction_horizon = 3.0  # [s]
        self.robot_is_stuck_velocity = 0.001
        # cost parameters
        self.goal_cost_weight = 0.15
        self.speed_cost_weight = 1.
        self.obstacle_cost_weight = 1.0
        # distance from robot to obstacle surface at which to ignore the obstacle
        # should be bigger than the robot radius
        self.obstacle_ignore_distance = 0.6

    def compute_cmd_vel(self, crowd, robot_pose, goal, show_plot=True, debug=False):
        x, y, th, vx, vy, w = robot_pose
        v = np.sqrt(vx*vx + vy*vy)

        # these params could be defined in init, or received from simulator
        human_radius = 0.3
        dt = 0.1  # time step [s]

        # Initialize the dynamic window
        # Velocity limits
        DWv_speed = [-self.robot_max_speed, self.robot_max_speed]
        DWv_rot = [-self.robot_max_w, self.robot_max_w]

        # Acceleration limits
        DWa_speed = [v - self.robot_max_accel * dt, v + self.robot_max_accel * dt]
        DWa_rot = [w - self.robot_max_w_dot * dt, w + self.robot_max_w_dot * dt]

        #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
        W_speed = [max(min(DWv_speed), min(DWa_speed)), min(max(DWv_speed), max(DWa_speed))]
        W_rot = [max(min(DWv_rot), min(DWa_rot)), min(max(DWv_rot), max(DWa_rot))]

        # initialize variables
        best_cost = np.inf
        worst_cost = -np.inf
        best_cmd_vel = [0.0, 0.0]
        best_trajectory = None

        if show_plot:
            from matplotlib import pyplot as plt
            plt.ion()
            plt.figure(1)
            plt.cla()

        # Sample window to find lowest cost
        # evaluate all trajectory with sampled input in dynamic window
        for speed in np.arange(min(W_speed), max(W_speed), self.speed_resolution):
            for rot in np.arange(min(W_rot), max(W_rot), self.rot_resolution):
                cmd_vel = (speed, rot)

                # Trajectory
                trajectory, cost = single_trajectory_and_costs(
                        robot_pose, goal, cmd_vel,
                        self.prediction_horizon, dt,
                        crowd, self.static_obstacles, human_radius, self.obstacle_ignore_distance,
                        self.robot_radius, self.robot_max_speed,
                        self.goal_cost_weight, self.speed_cost_weight, self.obstacle_cost_weight,
                )

                # Update best cost
                if cost <= best_cost:
                    best_cost = cost
                    best_cmd_vel = [speed, rot]
                    best_trajectory = trajectory

                if cost >= worst_cost:
                    worst_cost = cost

        if debug:
            for speed in np.arange(min(W_speed), max(W_speed), self.speed_resolution):
                for rot in np.arange(min(W_rot), max(W_rot), self.rot_resolution):
                    cmd_vel = (speed, rot)

                    # Trajectory
                    trajectory, cost = single_trajectory_and_costs(
                            robot_pose, goal, cmd_vel,
                            self.prediction_horizon, dt,
                            crowd, self.static_obstacles, human_radius, self.obstacle_ignore_distance,
                            self.robot_radius, self.robot_max_speed,
                            self.goal_cost_weight, self.speed_cost_weight, self.obstacle_cost_weight,
                    )

                    if debug:
                        plt.plot(trajectory[:, 0], trajectory[:, 1],
                                 c=plt.cm.viridis((cost - worst_cost) / (0.00001 + best_cost - worst_cost)))
                        plt.title("{}".format(cost))
                        plt.axis('equal')
                        plt.pause(0.001)

        if best_trajectory is None:
            best_trajectory = np.array([robot_pose])
            print("No solution found!")

        # spin if stuck
        if abs(best_cmd_vel[0]) < self.robot_is_stuck_velocity \
                and abs(best_cmd_vel[1]) < self.robot_is_stuck_velocity:
            best_cmd_vel[1] = max(W_rot)

        if show_plot:
            plt.ion()
            plt.figure(1)
            plt.cla()
            plt.gca().add_artist(plt.Circle((robot_pose[0], robot_pose[1]), self.robot_radius, color='b'))
            for id_, x, y in crowd:
                plt.gca().add_artist(plt.Circle((x, y), human_radius, color='r'))
            plt.plot(best_trajectory[:, 0], best_trajectory[:, 1], "-g")
            plt.plot(goal[0], goal[1], "xb")
            plt.axis("equal")
            plt.xlim([0, 22])
            plt.ylim([-5, 5])
            plt.pause(0.1)

        print("SOLUTION")
        print(best_cost)
        print(best_cmd_vel)

        return tuple(best_cmd_vel)

def single_trajectory_and_costs(robot_pose, goal, cmd_vel,
                                prediction_horizon, dt,
                                circle_obstacles, polygon_obstacles, human_radius, obstacle_ignore_distance,
                                robot_radius, robot_max_speed,
                                goal_cost_weight, speed_cost_weight, obstacle_cost_weight,
                                ):
    cmd_vel = np.array(cmd_vel, dtype=np.float32)
    trajectory = predict_trajectory(robot_pose, cmd_vel, prediction_horizon, dt)
    cost = total_cost(
        trajectory, goal,
        circle_obstacles, polygon_obstacles, human_radius, obstacle_ignore_distance,
        robot_radius, robot_max_speed,
        goal_cost_weight, speed_cost_weight, obstacle_cost_weight,
    )
    return trajectory, cost


def step_robot_dynamics(np.float32_t[:] pose,
                        np.float32_t[:] next_pose,
                        np.float32_t[:] cmd_vel,
                        np.float32_t dt):
    # in
    cdef np.float32_t x = pose[0]
    cdef np.float32_t y = pose[1]
    cdef np.float32_t th = pose[2]
    cdef np.float32_t vx = pose[3]
    cdef np.float32_t vy = pose[4]
    cdef np.float32_t w = pose[5]
    cdef np.float32_t speed = cmd_vel[0]
    cdef np.float32_t rot = cmd_vel[1]
    cdef np.float32_t next_x = 0.
    cdef np.float32_t next_y = 0.
    cdef np.float32_t next_th = 0.
    cdef np.float32_t next_vx = 0.
    cdef np.float32_t next_vy = 0.
    cdef np.float32_t next_w = 0.

    # first apply small rotation
    next_th = th + rot * dt
    # update velocities (no momentum)
    next_vx = speed * ccos(next_th)
    next_vy = speed * csin(next_th)
    next_w = rot
    # then move along new angle
    next_x = x + next_vx * dt
    next_y = y + next_vy * dt

    # out
    next_pose[0] = next_x
    next_pose[1] = next_y
    next_pose[2] = next_th
    next_pose[3] = next_vx
    next_pose[4] = next_vy
    next_pose[5] = next_w

def predict_trajectory(pose, cmd_vel, prediction_horizon, dt):
    # initialize
    timesteps = np.arange(0, prediction_horizon, dt)
    trajectory = np.zeros((len(timesteps)+1, pose.shape[0]), dtype=np.float32)
    trajectory[0] = pose * 1.
    # fill trajectory poses
    for i, time in enumerate(timesteps):
        step_robot_dynamics(trajectory[i], trajectory[i+1], cmd_vel, dt)

    return trajectory

def total_cost(trajectory, goal,
               circle_obstacles, polygon_obstacles, human_radius, obstacle_ignore_distance,
               robot_radius, robot_max_speed,
               goal_cost_weight, speed_cost_weight, obstacle_cost_weight,
               ):
    return (
        goal_cost_weight * goal_cost(trajectory, goal, robot_radius) +
        speed_cost_weight * speed_cost(trajectory, robot_max_speed) +
        obstacle_cost_weight * obstacles_cost(trajectory,
                                              circle_obstacles, human_radius,
                                              polygon_obstacles,
                                              robot_radius, obstacle_ignore_distance)
    )


def obstacles_cost(trajectory,
                   circle_obstacles, circle_radii,
                   polygon_obstacles,
                   robot_radius, obstacle_ignore_distance):

    # static obstacles
    polygons_cost = 0.
    for obs in polygon_obstacles:
        # TODO
        distances = np.array([np.inf])
        if np.any(distances <= robot_radius):
            polygons_cost = np.inf
        else:
            polygons_cost = 1. / distances

    # circular obstacles
    circulars_cost = 0.
    if len(circle_obstacles) > 0:
        xy_obstacles = circle_obstacles[:, 1:3]
        # dx, dy distance between every obstacle and every pose in the trajectory
        deltas = trajectory[:, None, :2] - xy_obstacles[None, :, :2] # [timesteps, obstacles, xy]
        # distance from obstacle surfaces
        distances = np.linalg.norm(deltas, axis=-1) - circle_radii
        min_dist = np.min(distances)
        if min_dist <= robot_radius:
            circulars_cost = np.inf
        elif min_dist > obstacle_ignore_distance:
            circulars_cost = 0
        else:
            circulars_cost = 1. / np.min(distances)

    return max(polygons_cost, circulars_cost)

def goal_cost(trajectory, goal, robot_radius):
    # looking only at the last step in the trajectory
    x, y, th, vx, vy, w = trajectory[-1]
    gx, gy = goal
    # cost increases if the final robot heading points away from the goal
    goal_heading = np.arctan2(gy - y, gx - x)  # heading of robot-goal vector
    robot_heading = th
    heading_difference_angle = angle_difference_rad(goal_heading, robot_heading)
    cost = np.abs(heading_difference_angle)

    # no cost if goal is reached
    if np.sqrt((gy - y)**2 + (gx - x)**2) <= robot_radius:
        cost = 0

    return cost

def speed_cost(trajectory, robot_max_speed):
    # 0 if robot goes at max speed, >0 linearly otherwise
    x, y, th, vx, vy, w = trajectory[-1]
    speed = np.sqrt(vx*vx + vy*vy)
    return robot_max_speed - speed
