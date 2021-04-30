#!/usr/bin/env python

import rosbag
import os
import cv2
from tqdm.auto import tqdm as tqdm
from cv_bridge import CvBridge

rosbag_location = './'
cv_bridge = CvBridge()


def timestamp_str(ts):
    t = ts.secs + ts.nsecs / float(1e9)
    return '{:.12f}'.format(t)

filenames = ["dvs_vpr_2020-04-21-17-03-03.bag",
             "dvs_vpr_2020-04-22-17-24-21.bag",
             "dvs_vpr_2020-04-24-15-12-03.bag",
             "dvs_vpr_2020-04-27-18-13-29.bag",
             "dvs_vpr_2020-04-28-09-14-11.bag",
             "dvs_vpr_2020-04-29-06-20-23.bag"]

for bagname in filenames:
    bagname = bagname.replace('.bag', '')
    with rosbag.Bag(rosbag_location+'/'+bagname+'.bag', 'r') as bag:
        topics = bag.get_type_and_topic_info().topics
        for topic_name, topic_info in topics.iteritems():
            if topic_name == '/dvs/image_raw':
                total_num_frames = topic_info.message_count
                print('Found {} frames in rosbag'.format(total_num_frames))

        if not os.path.isdir(rosbag_location+bagname+'/frames'):
            os.makedirs(rosbag_location+bagname+'/frames')

        with tqdm(total=total_num_frames) as pbar:
            for topic, msg, t in bag.read_messages(topics=['/dvs/image_raw']):
                if msg.header.stamp.secs < 100:
                    pbar.update(1)
                    continue

                cv_image = cv_bridge.imgmsg_to_cv2(msg, "bgr8")

                if cv_image.shape[0] == 260 and cv_image.shape[1] == 346:
                    cv2.imwrite(rosbag_location + bagname + '/frames/' + timestamp_str(msg.header.stamp) + '.png', cv_image)

                pbar.update(1)
