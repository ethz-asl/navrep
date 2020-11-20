from __future__ import print_function
import numpy as np
import os
from tqdm import tqdm
from pose2d import Pose2D, apply_tf

DOWNSAMPLE = None  # 17 (1080 -> 64 rays)
OLD_FORMAT = False

GOAL_REACHED_RADIUS = 0.5


def rosbag_to_lidar_dataset(
    bag_path="~/rosbags/test.bag", lidar_topics=["/combined_scan"]
):
    import rosbag
    bag_path = os.path.expanduser(bag_path)
    bag = rosbag.Bag(bag_path)

    print(bag_path)
    print("------------------------------------------------------------------------")

    scans = []
    for topic, msg, t in bag.read_messages(topics=lidar_topics):
        scans.append(msg.ranges)

    scans = np.array(scans)
    data = raw_scans_to_x(scans)
    return data


def rosbag_to_image_dataset(
    bag_path="~/rosbags/test.bag",
    image_topics=["/camera/color/image_raw"],
    archive_dir="~/navrep/datasets/V/im",
    resize_dim=(64, 64),
):
    import rosbag
    import cv2
    from cv_bridge import CvBridge
    from pyniel.python_tools.path_tools import make_dir_if_not_exists

    archive_dir = os.path.expanduser(archive_dir)

    bridge = CvBridge()

    bag_path = os.path.expanduser(bag_path)
    bag_name = os.path.basename(bag_path).replace(".", "_")
    bag = rosbag.Bag(bag_path)

    print(bag_path)
    print("------------------------------------------------------------------------")

    action_topics = ["/cmd_vel"]
    reward_topics = ["/goal_reached"]  # TODO: this is probably not getting published IRL
    images = []
    actions = []
    rewards = []
    dones = []
    current_action = [0, 0, 0]
    current_reward = -0.01
    goal_reward = 100.0
    for topic, msg, t in bag.read_messages(
        topics=image_topics + action_topics + reward_topics
    ):
        if topic in image_topics:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv_resized = cv2.resize(cv_image, resize_dim)
            images.append(cv_resized)
            actions.append(current_action)
            rewards.append(current_reward)
            dones.append(0)
        if topic in action_topics:
            current_action = [msg.linear.x, msg.linear.y, msg.angular.z]
        if topic in reward_topics:
            images.append(images[-1])
            actions.append(current_action)
            rewards.append(goal_reward)
            dones.append(1)
            break
    # if goal is not reached, make latest observation final
    if not dones[-1]:
        dones[-1] = 1
    # save
    images = np.array(images)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.array(dones)
    make_dir_if_not_exists(archive_dir)
    archive_path = os.path.join(
        archive_dir, "{}_images_actions_rewards_dones.npz".format(bag_name)
    )
    np.savez_compressed(
        archive_path, images=images, actions=actions, rewards=rewards, dones=dones
    )
    print("{} images found.".format(len(images)))
    print(archive_path, "written.")

    return images, actions, rewards, dones


def folder_to_lidar_dataset(
    directory="~/autoeval/2020_04_05_17_36", lidar_topics=["/combined_scan"]
):
    import rosbag
    directory = os.path.expanduser(directory)
    # list files
    bagfiles = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".bag")]:
            bagfiles.append(os.path.join(dirpath, filename))
    print("{} bag files found.".format(len(bagfiles)))
    scans = []
    for bag_path in bagfiles:
        bag = rosbag.Bag(bag_path)
        for topic, msg, t in bag.read_messages(topics=lidar_topics):
            scans.append(msg.ranges)
    print("{} scans found.".format(len(scans)))
    scans = np.array(scans)
    data = raw_scans_to_x(scans)
    return data


def msg_to_lidar_dataset(msg):
    """
    msg: LaserScan

    output: ndarray (n, N, channels), typically 0-100 [meters]
    """
    scans = np.array([msg.ranges])
    data = raw_scans_to_x(scans)
    return data


def x_to_raw_scans(data, msg):
    """
    upsamples if necessary

    data: ndarray (n, N, channels), typically 0-100 [meters]
    msg: LaserScan

    output: ndarray (n, n_rays in msg), typically 0-100 [meters]
    """
    N_RAYS = len(msg.ranges)
    if DOWNSAMPLE is not None:
        data = np.repeat(data, DOWNSAMPLE, axis=1)[:, :N_RAYS, :]
    ranges = np.zeros((len(data), N_RAYS,))
    ranges = data[:, :, 0]
    return ranges


def raw_scans_to_x(scans):
    """
    downsamples if necessary

    raw_scans: ndarray (n, scan_size), typically 0-100 [meters]

    output: ndarray (n, N, channels), typically 0-100 [meters]
    """
    if DOWNSAMPLE is not None:
        scans = scans[:, ::DOWNSAMPLE]
    N_RAYS = scans.shape[1]
    data = np.zeros((len(scans), N_RAYS, 1), dtype=np.float32)
    data[:, :, 0] = scans
    return data


def folder_to_archive(
    directory="~/autoeval",
    archive_dir="~/navrep/datasets/V/ian",
    lidar_topics=["/combined_scan"],
):
    import rosbag
    from pyniel.python_tools.path_tools import make_dir_if_not_exists
    # preproc args
    directory = os.path.expanduser(directory)
    archive_dir = os.path.expanduser(archive_dir)
    # list files
    bagfiles = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".bag")]:
            bagfiles.append(os.path.join(dirpath, filename))
    print("{} bag files found.".format(len(bagfiles)))
    if OLD_FORMAT:
        for i, bag_path in enumerate(bagfiles):
            scans = []
            bag = rosbag.Bag(bag_path)
            for topic, msg, t in bag.read_messages(topics=lidar_topics):
                scans.append(msg.ranges)
            scans = np.array(scans)
            data = raw_scans_to_x(scans)
            make_dir_if_not_exists(archive_dir)
            archive_path = os.path.join(
                archive_dir, "{:03}_{}rays.npy".format(i, data.shape[1])
            )
            np.save(archive_path, data)
            print(archive_path, "written.")
    # rewards
#     action_topics = ["/cmd_vel"]
    action_topics = ["/pepper_robot/odom"]
    reward_topics = ["/goal_reached"]
    goal_topics = ["/move_base_simple/goal"]
    topics = lidar_topics + action_topics + reward_topics + goal_topics
    for i, bag_path in enumerate(bagfiles):
        scans = []
        robotstates = []
        actions = []
        rewards = []
        dones = []
        current_action = np.array([0, 0, 0])
        current_reward = -0.01
        current_goal = None
        current_goal_frame = None
        goal_reward = 100.0
        print("Loading {}...".format(os.path.basename(bag_path)))
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
        for topic, msg, t in tqdm(bag.read_messages(topics=topics),
                                  total=bag.get_message_count(topic_filters=topics)):
            # update robotstate from current values
            current_goal_in_rob = current_goal
            if current_goal is not None and bag_transformer is not None:
                p2_cg_in_rob = Pose2D(bag_transformer.lookupTransform(
                    "base_footprint", current_goal_frame, msg.header.stamp))
                current_goal_in_rob = apply_tf(current_goal[None,:], p2_cg_in_rob)[0]
            robotstate = None
            if current_goal_in_rob is not None:
                robotstate = np.concatenate([current_goal_in_rob, current_action], axis=0)
            # process messages
            if topic in lidar_topics:
                if current_goal is None:
                    continue
                scan = np.array(msg.ranges)
                scans.append(scan * 1.)
                robotstates.append(robotstate * 1.)
                actions.append(current_action * 1.)
                rewards.append(current_reward * 1.)
                dones.append(0)
            if topic in action_topics:
#                 current_action = [msg.linear.x, msg.linear.y, msg.angular.z]  # if msg is cmd_vel
                current_action = np.array([  # if msg is odometry
                    msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z])
            if topic in reward_topics:
                if current_goal is None:
                    continue
                scans.append(scans[-1] * 1.)
                robotstates.append(robotstate * 1.)
                actions.append(current_action * 1.)
                rewards.append(goal_reward * 1.)
                dones.append(1)
                current_goal = None
            if topic in goal_topics:
                if bag_transformer is None:
                    continue
                goal_in_msg = np.array([msg.pose.position.x, msg.pose.position.y])
                p2_msg_in_rob = Pose2D(bag_transformer.lookupTransform(
                    "base_footprint", msg.header.frame_id, msg.header.stamp))
                new_goal = apply_tf(goal_in_msg[None,:], p2_msg_in_rob)[0]
                goal_within_reach = np.linalg.norm(new_goal) < GOAL_REACHED_RADIUS
                # reject new goal which is too close
                if goal_within_reach and current_goal is None:
                    continue
                # goal reached, reset goal
                elif goal_within_reach and current_goal is not None:
                    scans.append(scans[-1] * 1.)
                    robotstates.append(robotstate * 1.)
                    actions.append(current_action * 1.)
                    rewards.append(goal_reward * 1.)
                    dones.append(1)
                    current_goal = None
                    current_goal_frame = None
                # new far away goal -> set goal (episode start)
                elif not goal_within_reach and current_goal is None:
                    current_goal = goal_in_msg
                    current_goal_frame = msg.header.frame_id
                # goal change -> episode change
                elif not goal_within_reach and current_goal is not None:
                    scans.append(scans[-1] * 1.)
                    robotstates.append(robotstate * 1.)
                    actions.append(current_action * 1.)
                    rewards.append(0)
                    dones.append(1)
                    current_goal = goal_in_msg
                    current_goal_frame = msg.header.frame_id
        # if goal is not reached, make latest observation final
        if not dones[-1]:
            dones[-1] = 1
        # save
        scans = np.array(scans)
        robotstates = np.array(robotstates)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        make_dir_if_not_exists(archive_dir)
        archive_path = os.path.join(
            archive_dir, "{:03}_scans_robotstates_actions_rewards_dones.npz".format(i)
        )
        np.savez_compressed(
            archive_path, scans=scans, robotstates=robotstates, actions=actions, rewards=rewards, dones=dones
        )
        print(archive_path, "written.")


def archive_to_lidar_dataset(directory="~/navrep/datasets/V/ian", limit=None):
    directory = os.path.expanduser(directory)
    # list files
    files = []
    if OLD_FORMAT:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in [f for f in filenames if f.endswith(".npy")]:
                files.append(os.path.join(dirpath, filename))
        arrays = []
        for path in files:
            arrays.append(np.load(path))
        data = np.concatenate(arrays, axis=0)
    else:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in [
                f for f in filenames if f.endswith("scans_robotstates_actions_rewards_dones.npz")
            ]:
                files.append(os.path.join(dirpath, filename))
        arrays = []
        if limit is None:
            limit = len(files)
        for path in files[:limit]:
            arrays.append(np.load(path)["scans"])
        data = np.concatenate(arrays, axis=0)
    return data
