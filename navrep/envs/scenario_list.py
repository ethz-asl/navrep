import numpy as np
import os
from pkg_resources import resource_filename

scenarios = [
    "irosasl1",
    "irosasl2",
    "irosasl3",
    "irosasl4",
    "irosasl5",
    "irosasl6",
    "irosasl_office_j1",
    "irosasl_office_j2",
    "irosasl_office_j3",
    "irosasl_office_j4",
    "irosasl_office_j5",
    "irosasl_office_j6",
    "irosunity_scene_map1",
    "irosunity_scene_map2",
    "irosunity_scene_map3",
    "irosunity_scene_map4",
    "irosunity_scene_map5",
    "irosunity_scene_map6",
]

scenario_maps = [
    "asl",
    "asl",
    "asl",
    "asl",
    "asl",
    "asl",
    "asl_office_j",
    "asl_office_j",
    "asl_office_j",
    "asl_office_j",
    "asl_office_j",
    "asl_office_j",
    "unity_scene_map",
    "unity_scene_map",
    "unity_scene_map",
    "unity_scene_map",
    "unity_scene_map",
    "unity_scene_map",
]

rl_scenarios = [
    "rlasl1",
    "rlasl2",
    "rlasl3",
    "rlasl_office_j1",
    "rlasl_office_j2",
    "rlasl_office_j3",
    "rlunity_scene_map1",
    "rlunity_scene_map2",
    "rlunity_scene_map3",
]

rl_scenario_maps = [
    "asl",
    "asl",
    "asl",
    "asl_office_j",
    "asl_office_j",
    "asl_office_j",
    "unity_scene_map",
    "unity_scene_map",
    "unity_scene_map",
]

# simplest curriculum
# rl_scenarios = [ "rlasl2", ]
# rl_scenario_maps = [ "asl", ]

# hard curriculum
rl_scenarios = scenarios + rl_scenarios
rl_scenario_maps = scenario_maps + rl_scenario_maps

map_downsampling = {
    "empty": 0,
    "asl": 1,
    "asl_office_j": 3,
    "unity_scene_map": 2,
    "crowdmove1": 0,
    "crowdmove2": 0,
    "crowdmove3": 0,
    "crowdmove4": 0,
    "crowdmove5": 0,
    "crowdmove6": 0,
}

# TODO: define episode length per scenario

def set_scenario(args, scenario_name=None):
    if scenario_name is None:
        scenario_id = np.random.randint(len(scenarios))
    else:
        scenario_id = get_scenario_id(scenario_name)
    args.map_folder = resource_filename('asl_pepper_2d_sim_maps', 'maps')
    args.scenario = scenarios[scenario_id]
    args.map_name = scenario_maps[scenario_id]
    args.map_downsampling_passes = map_downsampling[args.map_name]

def set_rl_scenario(args, scenario_name=None):
    if scenario_name is None:
        scenario_id = np.random.randint(len(rl_scenarios))
    else:
        scenario_id = get_rl_scenario_id(scenario_name)
    args.map_folder = resource_filename('asl_pepper_2d_sim_maps', 'maps')
    args.scenario = rl_scenarios[scenario_id]
    args.map_name = rl_scenario_maps[scenario_id]
    args.map_downsampling_passes = map_downsampling[args.map_name]

def get_scenario_id(scenario_name):
    for i, name in enumerate(scenarios):
        if name == scenario_name:
            return i
    return None

def get_rl_scenario_id(scenario_name):
    for i, name in enumerate(rl_scenarios):
        if name == scenario_name:
            return i
    return None
