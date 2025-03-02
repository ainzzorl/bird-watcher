from ultralytics import YOLO

DETECTION_BATCH_SIZE = 10

def initialize_detection():
    return YOLO("yolo11n.pt")

def run_detection(model, input_files):
    result = []
    for (i, b) in enumerate(chunks(input_files, DETECTION_BATCH_SIZE)):
        print(f'Processing subbatch {i+1}')
        result += model(b)
    return result

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
