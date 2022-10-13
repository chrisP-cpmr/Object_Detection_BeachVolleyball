import cv2

# Convert to RGG - default of openCV is BGR - not needed in
rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Use a model to detect the location of the beach volleyball
##inputs = feature_extractor(images=frame, return_tensors="pt")
##outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
##target_sizes = torch.tensor([frames_height, frames_width])
##results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[count]

##for score, label, box in zip(results["scores"], results["lables"], results["boxes"]):
##    box = [round(i, 2) for i in box.tolist()]
# let0s only keep detections with score > 0.9
##    if score > 0.9:
##        print(
##            f"Detected {model.config.id2label[label.item()]} with confidence "
##            f"{round(score.item(), 3)} at location {box}"
##        )

# Loop through face locations array and draw a rectangle around each ball that is detected
##for top, right, bottom, left in ballLocation:
##cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)