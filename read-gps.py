import pynmea2

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


def smooth_gps(in_gps):
    N = 3
    out_lat = np.convolve(in_gps[:, 0], np.ones((N,)) / N, mode='valid')
    out_lon = np.convolve(in_gps[:, 1], np.ones((N,)) / N, mode='valid')
    out_t = np.convolve(in_gps[:, 2], np.ones((N,)) / N, mode='valid')
    return np.vstack((out_lat, out_lon, out_t)).T


vid_path_1 = 'traverse1.mp4'
vid_path_2 = 'traverse2.mp4'

x1 = get_gps(vid_path_1.replace(mov_file.split('.')[-1], 'nmea'))
x2 = get_gps(vid_path_2.replace(mov_file.split('.')[-1], 'nmea'))

match_x1_to_x2 = []
for idx1, (latlon, t) in enumerate(zip(x1[:, 0:2], x1[:, 2])):
    if len(match_x1_to_x2) < 6:
        min_idx2 = 0
        max_idx2 = int(0.25 * len(x2))
    elif idx1 > 0.5 * len(x1):
        min_idx2 = match_x1_to_x2[-5]
        max_idx2 = len(x2)
    else:
        min_idx2 = match_x1_to_x2[-5]
        max_idx2 = int(0.75 * len(x2))
    best_match = (np.linalg.norm(x2[min_idx2:max_idx2, 0:2] - latlon, axis=1)).argmin() + min_idx2
    # print('%d %.4f' % (idx1, np.linalg.norm(x2[best_match, 0:2] - latlon)))
    match_x1_to_x2.append(best_match)
match_x1_to_x2 = np.array(match_x1_to_x2)

t_raw1 = x1[:, 2]
t_raw2 = x2[match_x1_to_x2, 2]
timestamps_gps1 = np.array([t + video_beginning[traverse_to_name[comparison_id_no_suffix_traverse1]] for t in t_raw1])
timestamps_gps2 = np.array([t + video_beginning[traverse_to_name[comparison_id_no_suffix_traverse2]] for t in t_raw2])
