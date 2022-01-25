import cv2
import datetime
import numpy as np
from collections import defaultdict
from centroidtracker import CentroidTracker
import pandas as pd
import imutils


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
tracker = CentroidTracker(maxDisappeared=500, maxDistance=220)


def non_max_suppression_fast(boxes, overlapThresh):
	'"Cobine boundingboxes that overlap into one bbox"'
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

def convert_to_2d(Xcenter, y2):
	'" Convert the coordinates from a 3d playing field to a 2d playing field"'
	pts_src = np.array([[257, 262], [370, 225], [492, 190], [294, 324], [474, 272], [620, 213], [727, 383], [799, 259]])
	# Take points from the frame as reference and give the same point coordinates on the picture for a transformation
	pts_dst = np.array([[110, 145], [349, 145], [588, 145], [110, 500], [349, 500], [588, 500], [349, 855], [588, 855]])

	# calculate matrix H
	h, status = cv2.findHomography(pts_src, pts_dst)

	# provide a point you wish to map from image 1 to image 2
	a = np.array([[Xcenter, y2]], dtype='float32')
	a = np.array([a])

	# finally, get the mapping
	pointsOut = cv2.perspectiveTransform(a, h)
	pointsOut = pointsOut.astype(int)
	return pointsOut

def main(video="1639943552_6-967003_camera1_200-200-400-400_24_769.mp4"):
	'"Read the frames, recognise humans and track them. The coordinates of the bottom of the bbox are saved for transormation and plotting"'
	cap = cv2.VideoCapture(video)

	paddel_2d = cv2.imread('media/paddelfield.jpeg')

	fps_start_time = datetime.datetime.now()
	fps = 0
	total_frames = 0
	centroid_dict = defaultdict(list)
	object_id_list = []

	while True:
		ret, frame = cap.read()
		frame = imutils.resize(frame, width=800)
		# frame = frame[540:1080, 700:1920]
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

			# saving the converted cords
			pointsout = convert_to_2d(xCenter, y2)
			for tuple in pointsout:
				for points in tuple:
					pd.DataFrame({'x': [points[0]], 'y': [points[1]]}, index=[objectId]).to_csv('cords.csv', mode='a',
																								header=False)

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
			cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

		fps_end_time = datetime.datetime.now()
		time_diff = fps_end_time - fps_start_time
		if time_diff.seconds == 0:
			fps = 0.0
		else:
			fps = (total_frames / time_diff.seconds)
		fps_text = "FPS: {:.2f}".format(fps)
		cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

		# cv2.imshow("Application", frame)
		key = cv2.waitKey(1)
		if key == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()


main()



