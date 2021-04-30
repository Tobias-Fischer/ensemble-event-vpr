import pynmea2
import numpy as np


def get_gps(nmea_file_path):
    nmea_file = open(nmea_file_path, encoding='utf-8')

    latitudes, longitudes, timestamps = [], [], []

    first_timestamp = None
    previous_lat, previous_lon = 0, 0

    for line in nmea_file.readlines():
        try:
            msg = pynmea2.parse(line)
            if first_timestamp is None:
                first_timestamp = msg.timestamp
            if msg.sentence_type not in ['GSV', 'VTG', 'GSA']:
                # print(msg.timestamp, msg.latitude, msg.longitude)
                # print(repr(msg.latitude))
                dist_to_prev = np.linalg.norm(np.array([msg.latitude, msg.longitude]) - np.array([previous_lat, previous_lon]))
                if msg.latitude != 0 and msg.longitude != 0 and msg.latitude != previous_lat and msg.longitude != previous_lon and dist_to_prev > 0.0001:
                    timestamp_diff = (msg.timestamp.hour - first_timestamp.hour) * 3600 + (msg.timestamp.minute - first_timestamp.minute) * 60 + (msg.timestamp.second - first_timestamp.second)
                    latitudes.append(msg.latitude); longitudes.append(msg.longitude); timestamps.append(timestamp_diff)
                    previous_lat, previous_lon = msg.latitude, msg.longitude

        except pynmea2.ParseError as e:
            # print('Parse error: {} {}'.format(msg.sentence_type, e))
            continue

    return np.array(np.vstack((latitudes, longitudes, timestamps))).T
