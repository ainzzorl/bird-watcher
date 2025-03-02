#!/usr/bin/env python3

import argparse
import asyncio
import csv
import cv2
import datetime
import os
import signal
import subprocess
import sys
import threading
import time
import yaml

from telethon import TelegramClient

from classification import initialize_classifier, classify
from detection import initialize_detection, run_detection

config = None

root_directory = None
timelapse_directory = None
detections_directory = None
detections_csv_file = None

# TODO: comment
def wait_until_enough_space(dir, max_files):
    while True:
        files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        print(f'Num files in {dir}: {len(files)}')
        if len(files) < max_files:
            return
        print(f'Too many files in {dir}: {len(files)}. Waiting...')
        time.sleep(10)

def run_timelapse():
    global timelapse_process
    wait_until_enough_space(timelapse_directory, 1000)
    print('Starting timelapse')

    roi = []
    if config['roi']:
        roi = [
            '--roi-offset-x', str(config['roi']['offset_x']),
            '--roi-offset-y', str(config['roi']['offset_y']),
            '--roi-width', str(config['roi']['width']),
            '--roi-height', str(config['roi']['height']),
        ]
    timelapse_process = subprocess.Popen(
        [
            './timelapse.py',
            '--frames', '10',
            '--frame-rate', str(config['frame_rate']),
            *roi,
            timelapse_directory
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    def notify_when_done():
        global timelapse_process
        stdout, stderr = timelapse_process.communicate()
        print(f'Timelapse finished, status code: {timelapse_process.returncode}')
        timelapse_process = None
        print(f'Stdout:\n{stdout.decode()}')
        print(f'Stderr:\n{stderr.decode()}')
        run_timelapse()

    threading.Thread(target=notify_when_done).start()

# TODO: comment
def signal_handler(sig, frame):
    print('Interrupt received')
    if timelapse_process is not None:
        print('Terminating timelapse')
        try:
            timelapse_process.kill()
            print('Stopped timelapse')
        except Exception as e:
            print(f'Failed to stop timelapse: {e}')

    sys.exit(0)

# TODO: comment
most_recent_digest_at = None
most_recent_cluster_at = None

async def maybe_send_digest():
    global most_recent_digest_at
    global most_recent_cluster_at

    print('Maybe sending detections')
    if most_recent_digest_at is not None and (datetime.datetime.now() - most_recent_digest_at).total_seconds() <= int(config['digest']['min_interval_seconds']):
        print(f'Sent digest too recently ({most_recent_digest_at}), skipping')
        return

    detections = []
    with open(detections_csv_file.name, newline='') as detection_file:
        reader = csv.reader(detection_file, delimiter=',')

        for row in reader:
            detection_time = datetime.datetime.strptime(row[0], "%Y-%m-%d-%H-%M-%S")
            if (datetime.datetime.now() - detection_time).total_seconds() > 600:
                continue
            if most_recent_cluster_at is not None and detection_time <= most_recent_cluster_at:
                continue
            print(f"Detection time: {detection_time}, recent cluster at: {most_recent_cluster_at}")
            detections.append({
                'at': detection_time,
                'path': row[1],
                'detection_conf': float(row[3]),
                'class': row[4].title(),
                'class_conf': float(row[5])
            })

    if len(detections) == 0:
        print('Nothing to report')
        return

    most_recent_detection_age_seconds = (datetime.datetime.now() - detections[-1]['at']).total_seconds()
    if most_recent_detection_age_seconds < 180:
        print(f'Most recent detection is too recent ({most_recent_detection_age_seconds} seconds). Skipping for now')
        return

    detections.sort(key=lambda d: d['at'])

    first_in_cluster = len(detections) - 1
    while first_in_cluster > 0:
        if (detections[first_in_cluster]['at'] - detections[first_in_cluster - 1]['at']).total_seconds() >= 30:
            break

        first_in_cluster -= 1
    detections = detections[first_in_cluster:]
    print(f'Detections in cluster: {len(detections)}')

    max_conf = 0
    best_detection = None
    for detection in detections:
        if detection['detection_conf'] > max_conf:
            max_conf = detection['detection_conf']
            best_detection = detection
    async with TelegramClient('birds-session', config['telegram']['api_id'], config['telegram']['api_hash']) as tg:
        msg = f"Prediction: {best_detection['class']}, confidence: {best_detection['class_conf']}."
        print(msg)
        await tg.send_file(config['telegram']['destination_entity'], f'{detections_directory}/{best_detection["path"]}-raw.jpg', caption=msg)
        most_recent_digest_at = datetime.datetime.now()
        most_recent_cluster_at = detections[-1]['at']

processed_source_files = set()

MAX_BATCH_SIZE = 100

async def run_iteration(yolo):
    wait_until_enough_space(detections_directory, 1000)

    input_files = [f for f in os.listdir(timelapse_directory) if os.path.isfile(os.path.join(timelapse_directory, f)) and not f in processed_source_files]
    print(f'Num source: {len(input_files)}')

    start_time = time.time()
    # Exclude files that are too recent - they can be not fully saved yet
    input_files = [f for f in input_files if start_time - os.path.getmtime(os.path.join(timelapse_directory, f)) >= 10]
    input_files = input_files[:MAX_BATCH_SIZE]

    yolo_results = run_detection(yolo, [os.path.join(timelapse_directory, f) for f in input_files])
    print('Got detection results')

    for i in range(len(yolo_results)):
        f = input_files[i]
        processed_source_files.add(f)
        result = yolo_results[i]
        boxes = result.boxes
        cls_names = [yolo.names[c] for c in boxes.cls.numpy()]
        if 'bird' not in cls_names:
            continue
        print(f'Found bird in {f}')

        img = cv2.imread(os.path.join(timelapse_directory, f))
        file_created_at = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(timelapse_directory, f))).strftime("%Y-%m-%d-%H-%M-%S")

        had_bird = False
        for i in range(len(boxes.cls)):
            if yolo.names[boxes.cls.numpy()[i]] != 'bird':
                continue
            had_bird = True
            detection_conf = boxes.conf.numpy()[i]
            coord = boxes.xyxy.numpy()[i]

            padding = int(config['detection']['padding'])
            # TODO: handle out of bounds errors
            x1 = int(coord[0]) - padding
            x2 = int(coord[2]) + padding
            y1 = int(coord[1]) - padding
            y2 = int(coord[3]) + padding

            crop_img = img[y1:y2, x1:x2]
            cropped_path = os.path.join(detections_directory, f'{f}-detection-{i}.jpg')
            cv2.imwrite(cropped_path, crop_img)

            classification_results = classify(cropped_path)
            print(f'Classification results: {classification_results}')

            cls_str = ','.join([f'{cr[1]},{cr[2]:.4f}' for cr in classification_results])

            detections_csv_file.write(f'{file_created_at},{f},{i},{detection_conf:.4f},{cls_str}\n')
            detections_csv_file.flush()

        if had_bird:
            result.save(filename=os.path.join(detections_directory, f'{f}-boxes.jpg'))
            cv2.imwrite(os.path.join(detections_directory, f'{f}-raw.jpg'), img)

    for f in input_files:
        print(f'Deleting {f}')
        os.remove(os.path.join(timelapse_directory, f))

    await maybe_send_digest()

async def main():
    global root_directory

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory for saving results and intermediate files")
    args = parser.parse_args()

    root_directory = args.directory
    if not os.path.isdir(root_directory):
        sys.stderr.write(f'Directory not found: {root_directory}\n')
        exit(1)

    global config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    global timelapse_directory
    timelapse_directory = os.path.join(root_directory, 'timelapse')
    if not os.path.exists(timelapse_directory):
        os.makedirs(timelapse_directory)

    global detections_directory
    detections_directory = os.path.join(root_directory, 'detections')
    if not os.path.exists(detections_directory):
        os.makedirs(detections_directory)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    run_timelapse()

    global detections_csv_file
    detections_csv_file = open(os.path.join(root_directory, 'detections.csv'), 'a', encoding="utf-8")

    yolo = initialize_detection()

    initialize_classifier()
    print('Initializing classifier')

    index = 0
    while True:
        start_time = time.time()
        print(f'Iteration #{index}')
        await run_iteration(yolo)

        elapsed = time.time() - start_time
        to_wait = int(config['detection']['interval_seconds']) - elapsed
        if to_wait > 0:
            time.sleep(to_wait)
        index += 1

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
