# wuji-hand-description

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Release](https://img.shields.io/github/v/release/wuji-technology/wuji-hand-description)](https://github.com/wuji-technology/wuji-hand-description/releases)

Robot model description package for Wuji Hand. Provides URDF, MuJoCo (MJCF), and USD assets for simulation and visualization. Includes ROS2 launch and RViz configuration files for quick inspection of left and right hand models.

**Get started with [Quick Start](#quick-start). For detailed documentation, refer to [Wuji Hand Description Guide](https://docs.wuji.tech/docs/en/wuji-hand/latest/wuji-hand-description-guide/) on Wuji Docs Center.**

## Repository Structure

```text
├── launch/              // ROS2 launch files for model visualization
├── meshes/              // STL meshes for visual and collision geometry
├── mjcf/                // MuJoCo XML models for left and right hands
├── rviz/                // RViz presets for left and right hands
├── step/                // STEP source files for CAD exchange
├── usd/                 // Isaac Sim USD assets for left and right hands
├── urdf/                // URDF models for local tools and ROS2
├── CMakeLists.txt       // ROS2 package installation rules
├── package.xml          // ROS2 package manifest
└── README.md
```

## Quick Start

### Installation

```bash
git clone https://github.com/wuji-technology/wuji-hand-description.git
```

### Running

#### MuJoCo

```bash
# Navigate to the repository directory
cd wuji-hand-description

# Right hand
python -m mujoco.viewer --mjcf=mjcf/right.xml

# Left hand
python -m mujoco.viewer --mjcf=mjcf/left.xml
```

#### ROS2 and RViz

```bash
# Source ROS2 environment, replace <distro> with your installed ROS2 distribution
source /opt/ros/<distro>/setup.bash

# If not cloned in this workspace yet:
cd ~/ros2_ws/src
git clone https://github.com/wuji-technology/wuji-hand-description.git
cd ..

rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select wuji_hand_description
source install/setup.bash

# Left hand (default)
ros2 launch wuji_hand_description display.launch.py

# Right hand
ros2 launch wuji_hand_description display.launch.py hand:=right
```

#### Isaac Sim (USD)

Load `usd/left/wujihand.usd` or `usd/right/wujihand.usd` directly in Isaac Sim.
For a complete simulation example, see [isaaclab-sim](https://github.com/wuji-technology/isaaclab-sim).

## Contact

For any questions, please contact [support@wuji.tech](mailto:support@wuji.tech).
