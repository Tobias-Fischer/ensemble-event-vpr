# Event-Based Visual Place Recognition With Ensembles of Temporal Windows

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![HitCount](http://hits.dwyl.io/Tobias-Fischer/ensemble-event-vpr.svg)](./README.md)
[![stars](https://img.shields.io/github/stars/Tobias-Fischer/ensemble-event-vpr.svg?style=flat-square)](https://github.com/Tobias-Fischer/ensemble-event-vpr/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/Tobias-Fischer/ensemble-event-vpr?style=flat-square)](https://github.com/Tobias-Fischer/ensemble-event-vpr/issues)
[![GitHub repo size](https://img.shields.io/github/repo-size/Tobias-Fischer/ensemble-event-vpr.svg?style=flat-square)](./README.md)
<a href="https://qcr.github.io" alt="QUT Centre for Robotics Open Source"><img src="https://github.com/qcr/qcr.github.io/blob/master/misc/badge.svg?style=flat-square" /></a>

### License + Attribution
The RT-GENE code is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Commercial usage is not permitted. If you use this dataset or the code in a scientific publication, please cite the following [paper](http://doi.org/10.1109/LRA.2020.3025505) ([preprint and additional material](https://arxiv.org/abs/2006.02826)):

```
@article{fischer2020event,
  title={Event-Based Visual Place Recognition With Ensembles of Temporal Windows},
  author={Fischer, Tobias and Milford, Michael},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={4},
  pages={6924--6931},
  year={2020}
}
```

There is a dataset that accompanies this code repository: https://zenodo.org/record/4302805

Please note that we are still preparing the main chunks of code for public release.

Available code for the moment:
- The [correspondence-event-camera-frame-camera.py](./correspondence-event-camera-frame-camera.py) file contains the mapping between the rosbag names and the consumer camera video names. The variable `video_beginning` indicates the ROS timestamp within the bag file that corresponds to the first frame of the consumer camera video file.
- The [read-gps.py](./read-gps.py) file contains some helper functions to read in GPS data from the provided nmea files, and find matches between two traverses.

Please note that in our paper we used manually annotated and then interpolated correspondences; instead here we provide matches based on the GPS data. Therefore, the results between what is reported in the paper and what is obtained using the methods here will be slightly different.
