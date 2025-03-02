#!/usr/bin/python3

"""
Take photos and save to disk.
It must use the built-in Python to use picamera2.
"""

import argparse
import os
import sys
import time

from picamera2 import Picamera2  # type: ignore


def init_camera(roi):
    print("Initializing camera")

    picam2 = Picamera2()
    picam2.configure("still")
    picam2.start()

    # Give time for Aec and Awb to settle, before disabling them
    time.sleep(1)
    picam2.set_controls({"AeEnable": False, "AwbEnable": False, "FrameRate": 1.0})
    if roi is not None:
        picam2.set_controls(
            {
                "ScalerCrop": [
                    int(roi["offset_x"]),
                    int(roi["offset_y"]),
                    int(roi["width"]),
                    int(roi["height"]),
                ]
            }
        )
    # And wait for those settings to take effect
    time.sleep(1)
    return picam2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dest", help="destination")
    parser.add_argument("-f", "--frames", help="Frames before exiting", required=True)
    parser.add_argument("-r", "--frame-rate", help="Frame rate", required=True)
    parser.add_argument("--roi-width", help="ROI Width")
    parser.add_argument("--roi-height", help="ROI Height")
    parser.add_argument("--roi-offset-x", help="ROI Offset X")
    parser.add_argument("--roi-offset-y", help="ROI Offset Y")
    args = parser.parse_args()
    destination = args.dest
    num_frames = int(args.frames)
    frequency_seconds = 1 / float(args.frame_rate)

    if not os.path.isdir(destination):
        sys.stderr.write(f"Output directory not found: {destination}\n")
        exit(1)

    # TODO: check that either all ROI params are set or none
    roi = None
    if args.roi_width:
        roi = {
            "offset_x": args.roi_offset_x,
            "offset_y": args.roi_offset_y,
            "width": args.roi_width,
            "height": args.roi_height,
        }

    picam2 = init_camera(roi)

    index = 0
    while index < num_frames:
        start_time = time.time()
        r = picam2.capture_request()
        fn = time.strftime("%Y-%m-%d-%H-%M-%S") + ".jpg"
        r.save("main", os.path.join(destination, fn))
        r.release()
        print(f"Captured image {index}, {fn}")

        elapsed = time.time() - start_time
        to_wait = frequency_seconds - elapsed
        if to_wait > 0:
            time.sleep(to_wait)

        index += 1
