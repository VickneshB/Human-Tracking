import numpy as np
import cv2
import sys
import imutils
from imutils.object_detection import non_max_suppression
import time


def main():

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    video = cv2.VideoCapture(0)
    is_ok, bgr_image_input = video.read()

    if not is_ok:
        print("Cannot read video source")
        sys.exit()

    height = bgr_image_input.shape[0]
    width = bgr_image_input.shape[1]

    try:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fname = "OUTPUT.avi"
        fps = 30.0
        videoWriter = cv2.VideoWriter(fname, fourcc, fps, (width, height))
    except:
        print("Error: can't create output video: %s" % fname)
        sys.exit()

    fps = video.get(cv2.CAP_PROP_FPS)

    start = time.time()

    frame = 0
    while True:
        is_ok, bgr_image_input = video.read()
        if not is_ok:
            break

        frame = frame + 1
        # load the image and resize it to (1) reduce detection time
        # and (2) improve detection accuracy
        bgr_image_input = imutils.resize(bgr_image_input, width=min(400, bgr_image_input.shape[1]))
        orig = bgr_image_input.copy()
        # detect people in the image
        (rects, weights) = hog.detectMultiScale(bgr_image_input, winStride=(4, 4),
                                                padding=(8, 8), scale=1.05)
        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(bgr_image_input, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # show some information on the number of bounding boxes
        '''filename = imagePath[imagePath.rfind("/") + 1:]
        print("[INFO] {}: {} original boxes, {} after suppression".format(
            filename, len(rects), len(pick)))'''
        # show the output images
        #cv2.imshow("Before NMS", orig)

        now = time.time()
        fps = frame / (now-start)
        fps = np.round(fps, 2)
        cv2.putText(bgr_image_input, "fps: " + str(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("After NMS", bgr_image_input)

        videoWriter.write(bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

if __name__ == "__main__":
    main()