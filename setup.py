import numpy
from setuptools import setup
from Cython.Build import cythonize


setup(
    name="navrep",
    description='navigation representations',
    author='Daniel Dugas',
    version='0.0.17',
    # doesn't work with pip install -e
    package_dir={'frame_msgs': "external/frame_msgs", 'tf_bag': "external/tf_bag"},
    packages=["navrep",
              "navrep.tools",
              "navrep.envs",
              "navrep.models",
              "navrep.scripts",
              # external modules bundled in here for convenience
              "frame_msgs",
              "frame_msgs.msg",
              "tf_bag"],
    # packages=["navrep"],
    ext_modules=cythonize("crings/crings.pyx", annotate=True),
    python_requires='>=3.6',
    install_requires=[
        'numpy', 'matplotlib', 'Cython', 'rospkg', 'pyyaml', 'numba', # 'ipython', 'flake8', 'pylint',
        'rich',
        'tqdm',  # progress bar
        'pandas', # ==1.1.0', # for compatibility with saved files
        'stable-baselines', # ==2.10.0',
        'cloudpickle', # ==1.6.0', # try 1.3.0 if fails to load older C models
        'torch', # ==1.6.0',
        'torchvision', # ==0.3.0',  # used by CADRL, minGPT
        # 'tensorflow-gpu==1.13.2',
        'tensorflow', # ==1.13.2',
        'keras', # , ==2.3.1',
        'catkin_pkg', # needed by deep_social_planner
        'pyIAN', 'asl-pepper-2d-sim', 'soadrl-crowdnav',
        'pyniel', 'pymap2d', 'pylidar2d', 'pyrvo2-danieldugas', 'pyrangelibc-danieldugas',
    ],
    include_dirs=[numpy.get_include()],
    package_data={'navrep': ['maps/*']},
)
