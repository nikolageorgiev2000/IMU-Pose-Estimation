# Investigation of Pose Estimation From Sparse Inertial Data

For my final-year undergraduate dissertation at the University of Cambridge, I investigated the problem of inferring 3D pose from inertial sensor measurements. The implemented algorithms were successful at decreasing error in estimated pose compared to dead-reckoning on the tested IMU sensors as well as a synthetic dataset of human motion. A fixed-lag smoother was derived using Gaussian Belief Propagation on a junction tree and generalizes the established Kalman filter. Moreover, calibration of 9-axis IMUs was extensively researched to provide an end-to-end solution using techniques like ellipsoid fitting. This repository holds the modules I wrote for non-linear optimization, kinematic modeling, calibration, data collection and exact inference.

Full report: [pose_estimation.pdf](https://www.jtogen.com/texts/pose_estimation.pdf)

Many thanks to Andrea Ferlini for supervising me during this project, as well as Andreas Grammenos and Manon Kok for providing highly valuable advice regarding sensor calibration and fusion.
