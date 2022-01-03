import cv2
import datetime
import imutils
import numpy as np
from collections import defaultdict
from centroidtracker import CentroidTracker

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

# Only enable it if you are using OpenVino environment
# detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# maxDisappeared, time wait when object moves out of frame
tracker = CentroidTracker(maxDisappeared=700, maxDistance=220)


def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]


        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))


def main():
    cap = cv2.VideoCapture('1639943552_6-967003_camera1_200-200-400-400_24_769.mp4')

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    centroid_dict = defaultdict(list)
    object_id_list = []

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        total_frames = total_frames + 1
        (height, width) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (width, height), 127.5)
        detector.setInput(blob)
        person_detections = detector.forward()

        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.4)
        objects = tracker.update(rects)

        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            xCenter = int((x1 + x2) / 2)
            yCenter = int((y1 + y2) / 2)

            cv2.circle(frame, (xCenter, y2), 5, (0, 255, 0), -1)

            centroid_dict[objectId].append((xCenter, y2))

            if objectId not in object_id_list:
                object_id_list.append(objectId)
                start_pt = (xCenter, y2)
                end_pt = (xCenter, y2)
                cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
            else:
                L = len(centroid_dict[objectId])
                for pt in range(len(centroid_dict[objectId])):
                    if not pt + 1 == L:
                        start_pt = (centroid_dict[objectId][pt][0], centroid_dict[objectId][pt][1])
                        end_pt = (centroid_dict[objectId][pt + 1][0], centroid_dict[objectId][pt + 1][1])
                        cv2.line(frame, start_pt, end_pt, (0, 255, 0), 1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)
        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow("Application", frame)
        # cv2.VideoWriter_fourcc("new_vid.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


main()

