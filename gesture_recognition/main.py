import cv2
import matplotlib.pyplot as plt
import time
from bodypart_movement_verify import verify_arms_up, verify_legs_apart
video_capture = cv2.VideoCapture(0)        # 0 because it is capturing feed from main webcam linked to device

threshold = 0.001
# defining the connections between body parts to make a stick figure of human on webcam feed
point_connections = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7],[1,14],
                     [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]
while True:
    ret, frame = video_capture.read()
    time.sleep(4)

    # convert captured image frame into grey to make computation easier and efficient
    image_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_blob = cv2.dnn.blobFromImage(image = frame, scalefactor = 1.0 / 255, size = (256, 256))
    print(image_blob)

    # pre-trained weights for body part recognition
    network = cv2.dnn.readNetFromCaffe(
        'Weights/pose_deploy_linevec_faster_4_stages.prototxt',
        'Weights/pose_iter_160000.caffemodel')
    network.setInput(image_blob)
    output = network.forward()
    position_height = output.shape[2]
    position_width = output.shape[3]
    print(position_width, position_height)

    num_points = 15
    points = []
    for i in range(num_points):
        confidence_map = output[0, i, :, :]
        _, confidence, _, point = cv2.minMaxLoc(confidence_map)
        x = int((frame.shape[1] * point[0]) / position_width)
        y = int((frame.shape[0] * point[1]) / position_height)
        print(confidence)
        if confidence > threshold:
            cv2.circle(frame, (x, y), 5, (0,255,0), thickness = -1)
            cv2.putText(frame, "{}".format(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255))
            points.append((x, y))
        else:
            points.append(None)
        print(points)
    for connection in point_connections:
        partA = connection[0]
        partB = connection[1]
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (255,0,0))
        if verify_arms_up(points) == True and verify_legs_apart(points) == True:
            cv2.putText(frame, 'Complete', (50,200), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255))
            print(True)
        else:
            print(False)

    # will plot the movement on webcam feed shown in computer
    cv2.imshow('Video',frame)
    frame_number = 0
    # replace directory path as needed
    cv2.imwrite(f'/Users/krishhashia/Desktop/video_frame_results/gesture_results_{frame_number}.png', frame)
    frame_number += 1  # Increment frame number
    # to quit from webcam stream
    if cv2.waitKey(delay=2) & 0xFF=='Q':
        break