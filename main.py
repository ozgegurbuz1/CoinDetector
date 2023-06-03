import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob, os

toplam=0
model = cv2.dnn.readNetFromDarknet("yolov4.cfg","yolov4_last.weights")
layers = model.getLayerNames()
unconnect = model.getUnconnectedOutLayers()
unconnect = unconnect-1

output_layers = []
for i in unconnect:
    output_layers.append(layers[int(i)])


classFile = 'spot.names'
classNames=[]
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

img = cv2.imread('1.jpg')

img_width = img.shape[1]
img_height = img.shape[0]


img_blob = cv2.dnn.blobFromImage(img,1/255.0,(416,416),swapRB=True,crop=False)

model.setInput(img_blob)
detection_layers = model.forward(output_layers)

ids_list = []
boxes_list = []
confidences_list = []

for detection_layer in detection_layers:
    for object_detection in detection_layer:
        scores = object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence =scores[predicted_id]


        if confidence > 0.10:

            label = classNames[predicted_id]
            bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
            (box_center_x, box_center_y ,box_width ,box_height) = bounding_box.astype("int")
            start_x = int(box_center_x- (box_width/2))
            start_y = int(box_center_y - (box_height/2))

            ids_list.append(predicted_id)
            confidences_list.append(float(confidence))
            boxes_list.append([start_x,start_y,int(box_width),int(box_height)])

max_ids = cv2.dnn.NMSBoxes(boxes_list,confidences_list,0.5,0.4)


for max_id in max_ids:
    max_class_id=max_id
    box = boxes_list[max_class_id]

    start_x = box[0]
    start_y = box[1]
    box_width = box[2]
    box_height= box[3]

    predicted_id = ids_list[max_class_id]
    label = classNames[predicted_id]
    if(label=="1 lira"):
        toplam+=1
    elif(label=="50 kurus"):
        toplam+=(5/10)
    elif (label == "25 kurus"):
        toplam += (25/100)
    elif (label == "10 kurus"):
        toplam += (1/10)
    elif (label == "5 kurus"):
        toplam += (5/100)

    print(classNames[predicted_id]) #resimdeki sınıfların isimleri

    confidence=confidences_list[max_class_id]
    end_x = start_x + box_width
    end_y = start_y+box_height


    cv2.rectangle(img,(start_x,start_y),(end_x,end_y),(255, 0, 0),2)
    cv2.putText(img,label+"->"+str(round(confidence,2)),(start_x,start_y-1),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255, 0, 0),1,1)
    toplam = round(toplam, 4)  # virgülden sonra 4 basamak
cv2.putText(img,"Toplam :"+str(toplam) +" TL",(370,300),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0, 0, 255),2,1)
print("TOPLAM =",toplam,"TL")
cv2.imshow("img",img)
cv2.waitKey(0)