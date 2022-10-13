# pip install opencv-python
# pip install numpy
# pip install torch
# pip install transformers
# pip install pillow

import cv2
import os
import numpy as np
import time
from transformers import DetrFeatureExtractor, DetrForObjectDetection, pipeline, \
    YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image, ImageDraw



def get_metadata(file_path):
    """
    This function outputs the metadata of the input video file.

    :param file_path: Path to the input video file.
    :return: video_fps, total_frames, frames_height, frames_width
    """

    # read video from file
    recording = cv2.VideoCapture(file_path)

    # Read metadata of the video
    video_fps = recording.get(cv2.CAP_PROP_FPS)
    total_frames = int(recording.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_height = int(recording.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_width = int(recording.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"Frame per second: {video_fps} \nTotal Frames: {total_frames} \nHeight: {frames_height} \nWidth: {frames_width}")
    return video_fps, total_frames, frames_height, frames_width


def process_frames(file_path, output_file_path):
    """
    This function will take an input video and run every fram through the DETR-resnet-50 model to detect abjects in the frames.
    All objectes will be marked with a box and all frames will be put back together to the output video.

    :param file_path: Path to the input recording file
    :param output_file_path: Path and name of the output file
    :return: Returns the input video with boxes around the detected objects
    """

    # initialize variables
    count = 0                                   # count to ensure all frames have been processed
    list_frames = []                            # list of all processed frames
    recording = cv2.VideoCapture(file_path)     # reading the input video with cv2

    # extract video metadata
    video_fps = recording.get(cv2.CAP_PROP_FPS)                     # check fps of input video
    total_frames = int(recording.get(cv2.CAP_PROP_FRAME_COUNT))     # count total frames of input video
    frames_height = int(recording.get(cv2.CAP_PROP_FRAME_HEIGHT))   # check pixel height of input video frame
    frames_width = int(recording.get(cv2.CAP_PROP_FRAME_WIDTH))     # check pixel width of input video frame
    codec = cv2.VideoWriter.fourcc(*'mp4v')                         # define format of output video

    # define the video output format and the values to resize the frames
    output_frames_height = 1080
    output_frames_width = 1920
    video_writer = cv2.VideoWriter(output_file_path, codec, video_fps, (output_frames_width, output_frames_height))

    # Initialize detr-resnet-50 model
    ##feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")    # Taken from HuggingFace
    ##model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")              # Taken from HuggingFace

    # Initialize Yolos-tiny model
    feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')

    # create the object detection pipeline
    object_detection_pipe = pipeline("object-detection",
                                     model=model,
                                     feature_extractor=feature_extractor)

    while count != 50:



        # Read video and retrieve individual frames
        ret, frame = recording.read()

        if frame is None:
            continue

        # Resize image for faster processing
        resizeFrame = cv2.resize(frame, (output_frames_width, output_frames_height))

        # Convert np array to PIL image - pipeline only works with PIL images
        pilFrame = Image.fromarray(np.uint8(resizeFrame))

        start = time.perf_counter()

        # Detect all objects in the frame
        results = object_detection_pipe(pilFrame)

        end = time.perf_counter()

        ms = (end-start) * 10**6
        seconds = ms / (10**6)
        print(f"Elapsed {seconds:.03f} secs.")

        # Add boxes and description of boxes to image
        im1 = ImageDraw.Draw(pilFrame)
        for result in results:
            box = result['box']
            xmin, xmax, ymin, ymax = box['xmin'], box['xmax'], box['ymin'], box['ymax']
            label = result['label']
            prob = result['score']
            shape = [xmin, ymin, xmax, ymax]
            text = f'{label}: {prob:0.2f}'
            if label == "person":
                im1.rectangle(shape, outline="red", width=3)
                im1.text((xmin,ymax), text, fill="black")
            elif label == "sports ball":
                im1.rectangle(shape, outline="blue", width=3)
                im1.text((xmin, ymax), text, fill="black")
            else:
                continue

        # Convert PIL image back to numpy array
        pilFrame = np.array(pilFrame)

        # Write the frame to the output video
        video_writer.write(pilFrame)

        # Print for every 50 frames processed
        if (count % 50 == 0):
            print('Processed ', count, ' frames')
        list_frames.append(frame)
        if len(list_frames) == int(total_frames):
            break

        # Increase count for every processed frame
        count += 1

    # Release to close all the resources that we have opened for reading and writing the video
    recording.release()
    video_writer.release()

    cv2.destroyAllWindows()


# Define the input and output video location
dir = ('/Users/christophmeier/Code_MastersThesis/AttackHeigtTracker/inputVideo')
file = os.listdir(dir)[0]
path = str(dir) + '/' + str(file)

outputDir = '/Users/christophmeier/Code_MastersThesis/AttackHeigtTracker/'
outputFileName = 'processedVideo'
outputFileType = 'mp4'
outputFile = outputDir + outputFileName + '.' + outputFileType

# check metadata
get_metadata(path)

# process every frame
process_frames(path, outputFile)

