from __future__ import print_function
import numpy as np
import os
import rosbag
from tqdm import tqdm
from pose2d import Pose2D, apply_tf
from matplotlib import pyplot as plt
from CMap2D import CMap2D

from plot_ianenv_trajectories import blue, orange

DOWNSAMPLE = None  # 17 (1080 -> 64 rays)
OLD_FORMAT = False
ANIMATE = False

FIXED_FRAME = "gmap"
ROBOT_FRAME = "base_footprint"

GOAL_REACHED_DIST = 0.5


if __name__ == "__main__":
#     bag_path = os.path.expanduser("~/irl_tests/manip_corner_julian_jenjen.bag")
    bag_path = os.path.expanduser("~/irl_tests/hg_icra_round2.bag")

    cmdvel_topics = ["/cmd_vel"]
    joy_topics = ["/joy"]
    odom_topics = ["/pepper_robot/odom"]
    reward_topics = ["/patroller/reached_waypoint"]
    goal_topics = ["/move_base_simple/goal"]
    map_topics = ["/gmap"]
    topics = cmdvel_topics + joy_topics + odom_topics + reward_topics + goal_topics + map_topics

    bag_name = os.path.basename(bag_path)
    print("Loading {}...".format(bag_name))
    bag = rosbag.Bag(bag_path)
    try:
        import tf_bag
        bag_transformer = tf_bag.BagTfTransformer(bag)
    except ImportError:
        print("WARNING: Failed to import tf_bag. No goal information will be saved.")
        bag_transformer = None
        current_goal = [np.nan, np.nan]
    if bag.get_message_count(topic_filters=goal_topics) == 0:
        print("WARNING: No goal messages ({}) in rosbag. No goal information will be saved.".format(
            goal_topics))
        bag_transformer = None
        current_goal = [np.nan, np.nan]

    trajectories = []
    goals = []
    goals_reached = []
    trajectory = []
    goal = None
    i_frame = 0
    for topic, msg, t in tqdm(bag.read_messages(topics=topics),
                              total=bag.get_message_count(topic_filters=topics)):

        if topic in cmdvel_topics:
            cmd_vel = msg
            # store cmd_vel

        if topic in joy_topics:
            joy = msg
            # detect wether autonomous motion is active or not

        # process messages
        if topic in odom_topics:
            # velocity
            current_action = np.array([  # if msg is odometry
                msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z])

            # position
            try:
                p2_rob_in_fix = Pose2D(bag_transformer.lookupTransform(
                    FIXED_FRAME, ROBOT_FRAME, msg.header.stamp))
            except:  # noqa
                continue

            trajectory.append(p2_rob_in_fix)

        if topic in reward_topics:
            # goal is reached
            reached = True

        if topic in goal_topics:
            goal_in_msg = np.array([msg.pose.position.x, msg.pose.position.y])
            p2_msg_in_fix = Pose2D(bag_transformer.lookupTransform(
                FIXED_FRAME, msg.header.frame_id, msg.header.stamp))
            goal_in_fix = apply_tf(goal_in_msg[None, :], p2_msg_in_fix)[0]

            # end episode if goal change
            if goal is not None:
                if np.linalg.norm(goal - goal_in_fix) > GOAL_REACHED_DIST:
                    # goal change
                    trajectories.append(np.array(trajectory))
                    goals.append(goal)
                    goals_reached.append(np.any(
                        np.linalg.norm(np.array(trajectory)[:,:2] - goal, axis=1) < GOAL_REACHED_DIST))
                    trajectory = []

            # set goal
            goal = goal_in_fix

        if topic in map_topics:
            p2_map_in_fix = Pose2D(bag_transformer.lookupTransform(
                FIXED_FRAME, msg.header.frame_id, msg.header.stamp))
            mapmsg = msg

    fig, ax = plt.subplots(1, 1)

    if mapmsg is not None:
        map2d = CMap2D()
        map2d.from_msg(mapmsg)
        assert mapmsg.header.frame_id == FIXED_FRAME
        contours = map2d.as_closed_obst_vertices()
        for c in contours:
            cplus = np.concatenate((c, c[:1, :]), axis=0)
            ax.plot(cplus[:,0], cplus[:,1], color='k')
        plt.axis('equal')

    for t, g, s in zip(trajectories, goals, goals_reached):
        line_color = blue(len(t)/1000.)  # if s else orange(len(t)/1000.)
        zorder = 2 if s else 1
        if ANIMATE:
            yanim = np.ones_like(t[:,1]) * np.nan
            line, = ax.plot(t[:,0], yanim, color=line_color, zorder=zorder)
            ax.add_artist(plt.Circle((g[0], g[1]), 0.3, color="red", zorder=2))
            plt.pause(0.01)
            N = 10
            for i in range(0, len(yanim), N):
                yanim[i:i+N] = t[i:i+N,1]
                line.set_ydata(yanim)
                plt.pause(0.01)
                plt.savefig("/tmp/plot_irl_test_rosbags_{:05}.png".format(i_frame))
                i_frame += 1
        else:
            ax.plot(t[:,0], t[:,1], color=line_color, zorder=zorder)
            ax.add_artist(plt.Circle((g[0], g[1]), 0.3, color="red", zorder=2))
    ax.set_title(bag_name)
    ax.axis("equal")
    ax.set_adjustable('box')

    plt.show()
