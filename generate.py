import logging
import yaml
import torch

from synth_data_gen.segment_objects_with_boundaries import SegmentObjectsWithBoundaries


# log = logging.getLogger('my_logger')
# log.setLevel(logging.DEBUG)  # Set the log level

# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)

# log.addHandler(console_handler)

# file_handler = logging.FileHandler('log.log')
# file_handler.setLevel(logging.DEBUG)

# file_handler.setFormatter(formatter)
# log.addHandler(file_handler)


# # Log some messages
# # log.debug('This is a debug message')
# # log.info('This is an info message')
# # log.warning('This is a warning message')
# # log.error('This is an error message')
# # log.critical('This is a critical message')

# log.info("Starting new log")

# device = "cuda" if torch.cuda.is_available() else "cpu"

# with open('config.yaml', 'r') as file:
#     config = yaml.safe_load(file)


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"


    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    sg = SegmentObjectsWithBoundaries(config, device, verbose=True, parallel=True)
    vid = "/home/aiden/Documents/cs/SoundQ2/test/videoplayback (1).mp4"

    sg.test_video(vid)

test()