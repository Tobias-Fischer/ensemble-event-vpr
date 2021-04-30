import cv2
import numpy as np
from os.path import join, isfile, basename, abspath
import matplotlib.pyplot as plt


def compare_images(image_paths1, image_paths2, image_num1, image_num2):
    if isinstance(image_paths1[image_num1], np.ndarray):
        img_A = image_paths1[image_num1]
    else:
        img_A = cv2.imread(image_paths1[image_num1])

    if isinstance(image_paths2[image_num2], np.ndarray):
        img_B = image_paths2[image_num2]
    else:
        img_B = cv2.imread(image_paths2[image_num2])

    fig, ax = plt.subplots(1, 2, figsize=(2 * 7.2, 5.4))

    if isinstance(image_paths1[image_num1], str):
        ax[0].set_title(basename(image_paths1[image_num1]) + ' (Place ' + str(image_num1) + ')')
    else:
        ax[0].set_title('Place ' + str(image_num1))
    if isinstance(image_paths2[image_num2], str):
        ax[1].set_title(basename(image_paths2[image_num2]) + ' (Place ' + str(image_num2) + ')')
    else:
        ax[1].set_title('Place ' + str(image_num2))
    if len(img_A.shape) == 3 and img_A.shape[2] == 3:
        ax[0].imshow(img_A[..., ::-1])
    else:
        ax[0].imshow(img_A, 'gray', vmin=0, vmax=255)
    if len(img_B.shape) == 3 and img_B.shape[2] == 3:
        ax[1].imshow(img_B[..., ::-1])
    else:
        ax[1].imshow(img_B, 'gray', vmin=0, vmax=255)


def get_timestamps(comparison_folder1, comparison_folder2):
    def read_timestamps(comparison_folder):
        with open(join(comparison_folder, 'timestamps.txt'), "r") as file_in:
            timestamps_in = [float(line.rstrip()) for line in file_in]
        return timestamps_in
    return np.array(read_timestamps(comparison_folder1)), np.array(read_timestamps(comparison_folder2))


def get_timestamp_matches(timestamps, timestamps_to_match):
    timestamps_matched = np.array([np.abs(timestamps - ts).argmin() for ts in timestamps_to_match])
    return timestamps_matched


class ListOnDemand(object):
    def __init__(self):
        self.path_list = []
        self.image_list = []

    def append(self, item):
        self.path_list.append(item)
        self.image_list.append(None)

    def get_path_list(self):
        return self.path_list

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            #Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*idx.indices(len(self)))]
        elif isinstance(idx, list):
            return [self[ii] for ii in idx]
        elif isinstance(idx, np.ndarray):
            return self[idx.tolist()]
        elif isinstance(idx, str):
            return self[int(idx)]
        elif isinstance(idx, int) or isinstance(idx, np.generic):
            if idx < 0 : #Handle negative indices
                idx += len( self )
            if idx < 0 or idx >= len( self ) :
                raise IndexError("Index %d is out of range."%idx)

            if self.image_list[idx] is None:
                self.image_list[idx] = cv2.imread(self.path_list[idx])

            return self.image_list[idx]
        else:
            raise TypeError("Invalid argument type:", type(idx))

def get_image_sets_on_demand(image_paths1, image_paths2, matches1, matches2):
    images_set1 = ListOnDemand()
    images_set2 = ListOnDemand()
    for idx1 in range(len(matches1)):
        image_num1 = matches1[idx1]
        images_set1.append(image_paths1[image_num1])
    for idx2 in range(len(matches2)):
        image_num2 = matches2[idx2]
        images_set2.append(image_paths2[image_num2])
    return images_set1, images_set2


def get_vlad_features(sess, net_out, image_batch, images_set, save_name=None, batch_size=4, tqdm_position=1):
    if save_name and isfile(save_name):
        pre_extracted_features = np.load(save_name)
        if pre_extracted_features.shape[0] == len(images_set):
            return pre_extracted_features
        else:
            print('Warning: shapes of pre_extracted_features and images_set do not match, re-compute!')

    descs = []
    for batch_offset in tqdm(range(0, len(images_set), batch_size), position=tqdm_position, leave=False):
    # for batch_offset in range(0, len(images_set), batch_size):
        images = []
        for i in range(batch_offset, batch_offset + batch_size):
            if i == len(images_set):
                break

            image = images_set[i]
            if not isinstance(image, (np.ndarray, np.generic) ):
                image = cv2.imread(image)

            if image_batch.shape[3] == 1:  # grayscale
                images.append(np.expand_dims(np.expand_dims(image, axis=0), axis=-1))
            else:  # color
                image_inference = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(np.expand_dims(image_inference, axis=0))

        batch = np.concatenate(images, 0)
        descs = descs + list(sess.run(net_out, feed_dict={image_batch: batch}))

    netvlad_feature_list = np.array(descs)

    if save_name:
        np.save(save_name, netvlad_feature_list)

    return netvlad_feature_list
