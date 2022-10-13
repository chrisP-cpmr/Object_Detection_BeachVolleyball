# pip install opencv-python
# pip install numpy

import cv2
import os

dir = ('inputVideo/')
file = os.listdir(dir)[0]

# read video from file
recording = cv2.VideoCapture('/Users/christophmeier/Code_MastersThesis/AttackHeigtTracker/inputVideo/GH010950.MP4')  # (dir + '/'+ file)

# Read metadata of the video
video_fps = recording.get(cv2.CAP_PROP_FPS)
total_frames = int(recording.get(cv2.CAP_PROP_FRAME_COUNT))
height = int(recording.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(recording.get(cv2.CAP_PROP_FRAME_WIDTH))

print(f"Frame per second: {video_fps} \nTotal Frames: {total_frames} \nHeight: {height} \nWidth: {width}")

codec = cv2.VideoWriter.fourcc(*'mp4v')
video_writer = cv2.VideoWriter('output_video.mp4', codec, video_fps, (width, height))

# show first frame
##ret, frame = rec.read()

##cv2.imshow('',frame)
##cv2.waitKey(0)
##cv2.destroyAllWindows()


# Array to hold the locations where a ball was detected on individual frames
ballLocation = []

# Counter to keep track of the number of frames processed
count = 0

def showFrames(recording):
    while True:
        ret, frame = recording.read()
        if not ret:
            break # break if no next frame

        cv2.imshow('',frame) # show frame

        if cv2.waitKey(1) & 0xFF == ord('q'): # on press of q break
            break

        count =+ 1

    # release and destroy windows
    recording.release()    # close the window
    cv2.destroyAllWindows()     # De-allocate any associated memory usage

list_frames = []

while True: ##(count != total_frames):
    # Read video and retrieve individual frames
    ##print(str(count) + ' ' + str(total_frames))
    ret, frame = recording.read()

    # check 'ret' (return value) to see if we have read all the frames of the video an can exit loop
    #if not ret:
    if frame is None:
        continue
        ##print('All frames processed!')
        ##break

    # Convert to RGG - default of openCV is BGR
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use a model to detect the location of the beach volleyball -> insert a model from HF

    # Loop through face locations array and draw a rectangle around each ball that is detected
    ##for top, right, bottom, left in ballLocation:
        ##cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Write the frame to the output video
    video_writer.write(frame)

    # Print for every 50 frames processed
    if(count % 50 == 0):
        print('Processed ', count, ' frames')
    list_frames.append(frame)
    if len(list_frames) == int(total_frames):
        break

    count += 1

# Release to close all the resources that we have opened for reading and writing the video
recording.release()
video_writer.release()

cv2.destroyAllWindows()
