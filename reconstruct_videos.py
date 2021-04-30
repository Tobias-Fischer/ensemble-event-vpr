#!/usr/bin/env python

import os

filenames = ["dvs_vpr_2020-04-21-17-03-03.zip",
             "dvs_vpr_2020-04-22-17-24-21.zip",
             "dvs_vpr_2020-04-24-15-12-03.zip",
             "dvs_vpr_2020-04-27-18-13-29.zip",
             "dvs_vpr_2020-04-28-09-14-11.zip",
             "dvs_vpr_2020-04-29-06-20-23.zip"]

model = 'firenet_1000.pth.tar'
# model = 'E2VID_lightweight.pth.tar'

for filename in filenames:
    for num_events_per_pixel in [0.1, 0.3, 0.6, 0.8]:
        os.system("python run_reconstruction.py -c firenet_1000.pth.tar -i " + str(filename) + " --auto_hdr --color --num_events_per_pixel " + str(num_events_per_pixel) + " --hot_pixels_file " + str(filename.replace('.zip', '_hot_pixels.txt') + " --output_folder N_" + str(num_events_per_pixel) + "/" + str(filename.replace('.zip', '')) + " --dataset_name " + str(filename.replace('.zip', ''))))
    for window_duration in [44, 66, 88, 120, 140]:
        os.system("python run_reconstruction.py -c firenet_1000.pth.tar -i " + str(filename) + " --auto_hdr --color --window_duration " + str(window_duration) + " --hot_pixels_file " + str(filename.replace('.zip', '_hot_pixels.txt') + " --output_folder t_" + str(window_duration) + "/" + str(filename.replace('.zip', '')) + " --dataset_name " + str(filename.replace('.zip', ''))))
