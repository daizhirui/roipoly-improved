import sys
import os
import logging
import argparse
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
from my_roipoly import MultiRoi
import pickle

matplotlib.use('TkAgg')
logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                           '%(module)s.%(funcName)s:%(lineno)d %(message)s',
                    level=logging.INFO)


def display_help():
    print("Usage: python3 image_labeling [options] [image_dir]\n"
          "Options:\n"
          " --roi-type, -r roi_type_file    preload some roi types\n"
          " --output, -o pickle_file_name   specify the name of pickle file\n"
          " --mask-image, -m                output mask images\n"
          " --mask-image-only,              output mask images instead of a\n"
          "                                 pickle file"
          )


class GUI:
    def __init__(self, image_dir: str, roi_types=()):
        # load images
        self.image_dir = image_dir
        self.image_dict = dict()
        if os.path.isdir(image_dir):
            image_names = sorted(os.listdir(image_dir))
            image_names.remove('.DS_Store')
            cnt = 0
            logging.log(logging.INFO, "Loading images ...")
            for image_name in tqdm(image_names):
                image_path = image_dir + '/' + image_name
                if os.path.isfile(image_path):
                    cnt += 1
                    self.image_dict[image_name] = plt.imread(image_path)
            logging.log(logging.INFO, "{} images are loaded.".format(cnt))
        else:
            logging.log(logging.WARNING,
                        "{} is not a folder!".format(image_dir))
            exit(1)
        self.unprocessed_images = sorted(list(self.image_dict.keys()))
        self.mask_dict = {}
        self.current_image = self.unprocessed_images.pop()
        self.multi_roi = MultiRoi(img=self.image_dict[self.current_image],
                                  roi_types=roi_types,
                                  finish_callback=self.multiroi_finish_callback)

    def multiroi_finish_callback(self, multi_roi):
        self.mask_dict[self.current_image] = multi_roi.mask
        if len(self.unprocessed_images) > 0:
            self.current_image = self.unprocessed_images.pop()
            logging.log(logging.INFO, "Next img: {}".format(self.current_image))
            multi_roi.image = self.image_dict[self.current_image]
            # tell multi_roi continue process images
            return False
        else:
            # tell multi_roi no more image to process
            return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label image ROIs')
    parser.add_argument('--roi-type', dest='roi_types', required=False,
                        help='preload some roi types, separated by ","')
    parser.add_argument('--output', dest='pickle_name', default='mask.pickle',
                        help='the name of pickle file contains mask')
    parser.add_argument('image_dir',
                        help='the directory of images to be labelled')
    args = parser.parse_args()
    # check image folder path
    if os.path.exists(args.image_dir):
        image_dir = os.path.abspath(args.image_dir)
        gui = GUI(args.image_dir,
                  () if args.roi_types is None else args.roi_types.split(','))
        # image labeling ended, now save the mask
        with open(args.pickle_name, 'wb') as f:
            pickle.dump((gui.mask_dict, gui.multi_roi.roi_values), f)
    else:
        logging.log(logging.ERROR, "{} doesn't exist!".format(args.image_dir))
        exit(1)
