from keras.models import load_model
import numpy as np 
import cv2


mask_model = load_model("./model/97_model.hdf5")

prototxt_path = "./model_data/deploy.prototxt"
caffemodel_path = "./model_data/weights.caffemodel"


font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 1
color = (255, 0, 0) 
thickness = 2


# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
cap = cv2.VideoCapture(0)


def image_preprocess(img):
	resize_img = cv2.resize(img, (150, 150))
	resize_img = resize_img/255
	resize_img = np.reshape(resize_img, (1, *resize_img.shape))
	return resize_img


while(cap.isOpened()):

	ret, image = cap.read()
	if not image is None:

		(h, w) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

		model.setInput(blob)
		detections = model.forward()

		# Identify each face
		for i in range(0, detections.shape[2]):
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			confidence = detections[0, 0, i, 2]

			if (confidence > 0.5):
				tolerance = 5
				face_crop_image = image[startY - tolerance:endY + tolerance, startX - tolerance:endX + tolerance]
				image = cv2.rectangle(image,(startX - tolerance,startY - tolerance),(endX + tolerance,endY + tolerance),(0,255,0),3)

				if face_crop_image.shape[0] > 150  and face_crop_image.shape[1] > 150:
					preprocess_image = image_preprocess(face_crop_image)
					class_prediction = mask_model.predict(preprocess_image)[0][0]

					if class_prediction > 0.5:
						image = cv2.putText(image, 'No Mask', (startX - tolerance,startY - tolerance), font,  
		                   fontScale, (0, 0, 255) , thickness, cv2.LINE_AA) 
					else:
						image = cv2.putText(image, 'Mask', (startX - tolerance,startY - tolerance), font,  
		                   fontScale, color, thickness, cv2.LINE_AA) 
					

		cv2.imshow('frame',image)

	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
