# ASL internal script, automates installing all deps as editable (development)
sudo apt-get install -y python-dev python3-dev cmake virtualenv
# Create and source a virtualenv
cd ~
rm -r pepper3venv
set -e  # stop in case of failure
set -x  # print commands being executed
virtualenv pepper3venv --python=python3.6
source ~/pepper3venv/bin/activate

# necessary for pip install to work
pip install Cython numpy

# pip dependencies (now in setup.py)
pip install numpy matplotlib Cython rospkg pyyaml numba ipython flake8 pylint rich
pip install tqdm  # progress bar
pip install pandas==1.1.0 # for compatibility with saved files
pip install stable-baselines==2.10.0 cloudpickle==1.6.0 # try 1.3.0 if fails to load older C models
pip install torch==1.6.0 torchvision==0.3.0  # used by CADRL, minGPT
pip install tensorflow-gpu==1.13.2
pip install keras==2.3.1 catkin_pkg # needed by deep_social_planner

# if CLUSTER
# # pip install --user torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html  # for the cluster
# # pip install --user tensorflow==1.13.2  # for the cluster

cd ~/Code/navrep
pip install -e .

# SOADRL
cd ~/Code/deep_social_planner
pip install -e .
mkdir -p ~/soadrl/configs
cp ~/Code/deep_social_planner/scripts/A3C_navigation/crowd_nav/config/test_soadrl_static.config ~/soadrl/configs/
# asl_pepper_2d_simulator
cd ~/Code/pepper_ws/src/asl_pepper/asl_pepper_2d_simulator/python
pip install -e .
# IAN
cd ~/Code/pepper_ws/src/interaction_actions/python
pip install -e .
# responsive
cd ~/Code/pepper_ws/src/responsive/lib_dwa
pip install -e .
cd ~/Code/pepper_ws/src/responsive/lib_clustering
pip install -e .

# Python dependencies
cd ~/Code/pyniel
pip install -e .
cd ~/Code/pepper_ws/src/range_libc/pywrapper
python setup.py install
cd ~/Code/pepper_ws/src/pymap2d
pip install -e .
cd ~/Code/pepper_ws/src/pylidar2d
pip install -e .
cd ~/Code/pepper_ws/src/Python-RVO2
pip install -e .

# ros python dependencies
# cp ~/Code/navrep/external/frame_msgs ~/pepper3venv/lib/python3.6/site-packages/ -r
# cp ~/Code/navrep/external/tf_bag ~/pepper3venv/lib/python3.6/site-packages/ -r  # used in make_vae_dataset --env irl
pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag tf tf2_ros

# if CLUSTER:
# # cp ~/Code/navrep/external/frame_msgs ~/.local/lib64/python3.6/site-packages/ -r
# # cp ~/Code/navrep/external/tf_bag ~/.local/lib64/python3.6/site-packages/ -r

# maps
ln -s -i ~/Code/navrep/maps ~/maps
