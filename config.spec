specVersion: 1
description: >-
  Pure-sensor extensible RGB-D visual block perception, local costmap approach,
  and container interaction (inspect / withdraw / deposit) service.

properties:
  rgb_topic:
    type: string
    default: /minecraft/camera/rgb
    description: ROS2 RGB Image topic.

  depth_topic:
    type: string
    default: /minecraft/camera/depth
    description: ROS2 Depth Image topic (32FC1 or 16UC1).

  intrinsics_topic:
    type: string
    default: /minecraft/camera/intrinsics
    description: ROS2 CameraInfo topic supplying pinhole K matrix.

  extrinsics_topic:
    type: string
    default: /minecraft/camera/extrinsics
    description: ROS2 TransformStamped topic from base frame to camera optical frame.

  odom_topic:
    type: string
    default: /minecraft/chassis/odom
    description: ROS2 Odometry topic in minecraft_odom.

  local_cost_grid_topic:
    type: string
    default: /robonix/local_mapping/cost_grid
    description: ROS2 OccupancyGrid from local_mapping for collision-free approach pose selection.

  cmd_vel_topic:
    type: string
    default: /cmd_vel
    description: ROS2 Twist topic used during gentle 360-degree room scanning.

  use_sim_time:
    type: boolean
    default: true
    description: Use /clock for ROS2 timestamps.

  blocks_config_file:
    type: string
    default: config/blocks.yaml
    description: Declarative YAML registry of detectable block specs (size bounds, MC IDs, detector backend).

  detector_model_path:
    type: string
    default: ""
    description: Optional path to a multi-class ONNX object detector (YOLO/RT-DETR); empty uses classic CV plugins.

  detect_rate_hz:
    type: number
    minimum: 0.5
    maximum: 15.0
    default: 4.0
    description: Maximum RGB-D detection rate in Hz.

  min_confidence:
    type: number
    minimum: 0.1
    maximum: 0.99
    default: 0.45
    description: Minimum detection confidence score.

  max_detect_range_m:
    type: number
    minimum: 2.0
    maximum: 20.0
    default: 10.0
    description: Maximum depth backprojection distance in metres.

  min_track_hits:
    type: integer
    minimum: 1
    maximum: 10
    default: 2
    description: Minimum multi-frame observations required to confirm a detected 3D block.

  webui_enabled:
    type: boolean
    default: true
    description: Enable the live RGB detection preview and container test Web UI.

  webui_host:
    type: string
    default: 127.0.0.1
    description: Bind host for the debug Web UI.

  webui_port:
    type: integer
    minimum: 0
    maximum: 65535
    default: 8094
    description: Bind port for the debug Web UI; 0 disables.
