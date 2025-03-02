#!/usr/bin/python3

import argparse
import os
import sys
import time

from picamera2 import Picamera2 # type: ignore

# TODO: to config
ROI_OFFSET = (2250, 1300)
ROI_SIZE = (800, 400)

MAX_PHOTOS = 1000
RESTART_AFTER = 1000

def wait_until_enough_space(destination):
    while True:
        files = [f for f in os.listdir(destination) if os.path.isfile(os.path.join(destination, f))]
        print(f'Num files: {len(files)}')
        if len(files) < MAX_PHOTOS:
            return
        print(f'Too many files: {len(files)}. Waiting...')
        time.sleep(10)

def init_camera():
    print('Initializing camera')
    picam2 = Picamera2()
    picam2.configure("still")
    picam2.start()

    # Give time for Aec and Awb to settle, before disabling them
    time.sleep(1)
    picam2.set_controls({"AeEnable": False, "AwbEnable": False, "FrameRate": 1.0})
    picam2.set_controls({"ScalerCrop": [2250, 1300, 800, 400]})
    # And wait for those settings to take effect
    time.sleep(1)
    return picam2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dest", help="destination")
    parser.add_argument('-f', '--frames', help='Frames before exiting', required=True)
    parser.add_argument('-r', '--frame-rate', help='Frame rate', required=True)
    args = parser.parse_args()
    destination = args.dest
    num_frames = int(args.frames)
    frequency_seconds = 1 / float(args.frame_rate)

    if not os.path.isdir(destination):
        sys.stderr.write(f'Output directory not found: {destination}\n')
        exit(1)

    picam2 = init_camera()

    index = 0
    while index < num_frames:
        start_time = time.time()
        r = picam2.capture_request()
        fn = time.strftime('%Y-%m-%d-%H-%M-%S') + '.jpg'
        r.save("main", os.path.join(destination, fn))
        r.release()
        print(f"Captured image {index}, {fn}")

        # wait_until_enough_space(destination)

        elapsed = time.time() - start_time
        to_wait = frequency_seconds - elapsed
        if to_wait > 0:
            time.sleep(to_wait)

        index += 1
        if index % RESTART_AFTER == 0:
            print(f'Reinitializing camera after {index} frames')
            picam2.stop()
            time.sleep(1)
            picam2.close()
            time.sleep(1)
            picam2 = init_camera()
