set -x  # print commands being executed
mkdir ~/Code/pepper_ws/src -p

cd ~/Code 
git clone git@github.com:danieldugas/pyniel.git
git clone git@github.com:danieldugas/deep_social_planner.git
cd ~/Code/pepper_ws/src
git clone git@github.com:ethz-asl/asl_pepper.git --branch devel
git clone git@github.com:danieldugas/range_libc.git --branch comparisons
git clone git@github.com:danieldugas/pymap2d.git
git clone git@github.com:danieldugas/pylidar2d.git
git clone git@github.com:ethz-asl/interaction_actions --branch devel
git clone git@github.com:ethz-asl/pepper_local_planning.git responsive --branch asldemo
git clone git@github.com:danieldugas/Python-RVO2.git

cd ~/Code/pyniel && git pull
cd ~/Code/deep_social_planner && git pull
cd ~/Code/navrep && git pull
cd ~/Code/pepper_ws/src/asl_pepper && git pull
cd ~/Code/pepper_ws/src/range_libc && git pull
cd ~/Code/pepper_ws/src/pymap2d && git pull
cd ~/Code/pepper_ws/src/pylidar2d && git pull
cd ~/Code/pepper_ws/src/interaction_actions && git pull
cd ~/Code/pepper_ws/src/responsive && git pull
cd ~/Code/pepper_ws/src/Python-RVO2 && git pull
