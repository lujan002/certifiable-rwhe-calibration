# Fork of certifiable-rwhe-calibration for Northeastern EECE 5550: Mobile Robotics Final Project
Original repository: https://github.com/utiasSTARS/certifiable-rwhe-calibration

Our work extends the certifiably
correct algorithm for generalized robot-world and hand-eye
calibration (RWHEC) presented in the above repository by adding scripts and instructions for data processing, synchronization, and results vizualization for a simplified, mutli-sensor robotic platform. Our data consists of monocular camera videos (with apriltag detections) and optitrack motion capture data. 
The intention behind this project is to see whether this method can be implemented on a simplified platform with two cameras.

<img width="345" height="279" alt="problem statement" src="https://github.com/user-attachments/assets/e5a72a3a-061b-4b93-9fb5-fb1744bee7ea" />
<img width="200" height="300" alt="problem statement" src="https://github.com/user-attachments/assets/fc1b88ca-e6f8-4f0d-bae3-d93402b22516" />
<img width="400" height="400" alt="problem statement" src="https://github.com/user-attachments/assets/e56900fc-df42-490c-a4af-f3e1c687cd3e" />


# Step-by-Step Instructions
## Cut Video

Use your video editor of choice to find the precise timestamp (ms) where the flip occured. I used Lossless Cut. After writing down timestamps, run the following for each video:

`ffmpeg -i input.mp4 -ss 00:01:30 -t 00:01:15 -c copy output.mp4`
where -ss: start time and -t: duration 

ex: `ffmpeg -i Left_success3.mp4 -ss 00:00:01.342 -t 00:00:55 -c:v libx264 -c:a aac Left_success3_cut.mp4`

Below are the specific camera time offsets I used for a selection of the data (where the video was cut, aka where the “sync flip” occurs):

Success1

- Cam 1: 3.541
- Cam 2: 1.416

Success3

- Cam 1: 1.342
- Cam 2: 3.907

## Cut Optitrack Data

Run this to plot the euler angles (roll, pitch, yaw) of the optitrack data (world-to-camera rig poses):

`python3 plot_optitrack.py data/high-bay/raw/optitrack_success3.csv` 

Once plots show up, zoom in on the spikes that occur during the sync flips, record the time (x axis) and delete up to those collumns in the csv:

Ex: optitrack_success3: trim at 1.2s

`python3 cut_optitrack.py data/high-bay/raw/optitrack_success3.csv 1.2`

## Visualize Optitrack Pose Trajectory

`python3 visualize_optitrack_pose.py data/high-bay/raw/optitrack_success3_cut.csv`

## Construct B

To construct the matrix of camera-to-tag poses, run the following:

```jsx
cd ~/Desktop/certifiable-rwhe-calibration
python3 construct_B.py "data/high-bay/raw/video/Left_success3_cut.mp4" \
--num-samples 200 \
--show-visualization \
--tag-family tag36h11 \
--camera-matrix cam_1_intrinsics.yaml
```

(repeat for right/cam2 - don’t forget to change intrinsics path to cam2!)

**`Options:`**

- --output: Specify output CSV path (default: {video_name}_poses.csv)
- --tag-size: Tag size in meters (default: 0.15)
- --tag-id: Track a specific tag ID (default: uses first detected tag)
- --camera-matrix: Path to camera calibration file YAML
- --num-samples: Process frames at a specific rate (e.g., --sample-fps 1.0 for 1 frame/second)
- --show-visualization: Show visualization playback with detected AprilTags overlaid
- --tag-family: Default tag36h11

The CSV will include:

- timestamp: Time in seconds
- tag_id: ID of the AprilTag that was detected at that timestep (can have multiple rows each with unique tag_id for each timestep)
- qx, qy, qz, qw: Quaternion orientation
- x, y, z: Position in meters

## Combined Visualization

```jsx
python3 visualize_combined_poses.py \
--optitrack data/high-bay/raw/optitrack_success3_cut.csv \
--detection-csv data/high-bay/raw/Left_success3_cut.csv \
--video data/high-bay/raw/video/Left_success3_cut.mp4 \
--camera-matrix cam_1_intrinsics.yaml \
--start-frame 205
```

## Construct A-B pairs

Now once A and B csv files are made, we need to input into `construct_synchronized_A_and_B.py` which perfectly syncs timestamps (greedy policy to find where time-steps best match both datasets and complies new csv pair where each row is a matched / filtered row from the orignal two datasets)

```jsx
  python3 construct_synchronized_A_and_B.py \
    --csv-A data/high-bay/raw/optitrack_success3_cut.csv \
    --camera 1:data/high-bay/raw/Left_success3_cut.csv \
    --camera 2:data/high-bay/raw/Right_success3_cut.csv \
    --output data/high-bay/combined \
    --threshold 0.1
```

## Run Main Julia Solver

`julia --project=. experiments/rw_multi_eye_multi_hand_copy.jl`

Apparently if a tag only appears with one camera, there can be a singular (non-invertable) matrix error (i.e. tag 8 on success1 data). 

quick fix:    `rm data/high-bay/combined/tag_8_*`

## Visualize tag poses (Y)

It will be helpful to run construct_B again just to verity with the live AprilTag viewer that pops if the tags are in the right spot

`python3 plot_tag_poses.py`

`python3 plot_tag_poses.py tag_poses.csv data/high-bay/raw/optitrack_success3_cut.csv`
