# Reference pose tasks

This directory contains components to define tasks based on reference poses (e.g
motion capture data) as well as a motion capture tracking tasks. The tasks and
associated utils were developed as part of
[CoMic: Complementary Task Learning & Mimicry for Reusable Skills (2020)][hasenclever2020].

The reference data is stored in HDF5 files, which can be loaded using the
`HDF5TrajectoryLoader` class in `dm_control/locomotion/mocap/loader.py`. To
download the data used in the CoMic project, please use
`dm_control/locomotion/mocap/cmu_mocap_data.py`. In the reference pose tasks,
reference trajectories are represented as `Trajectory` objects (see
`dm_control/locomotion/mocap/trajectory.py`). For an example of how to construct
a task, see `dm_control/locomotion/examples/cmu_2020_tracking.py`.

[hasenclever2020]: https://proceedings.icml.cc/static/paper_files/icml/2020/5013-Paper.pdf
