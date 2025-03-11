from ultralytics import YOLO
from PIL import Image, ImageDraw

image_path = '/Users/kcoolyt/Desktop/Khalid/Research/pythonProject1/fireimage1.JPG'
FOV = 70
# Load the YOLOv8 model
model = YOLO('/Users/kcoolyt/Downloads/best-6.pt')

# Perform object detection on the image
results = model.predict(source=image_path , conf=0.25,save=True)

# Get the bounding box coordinates of the detected objects
#coordinates = results.pred[0][:, :4]

# Print the coordinates
#print(coordinates)
# Perform object detection on the image

for result in results:
    boxes = result.boxes
    bbox = boxes.xyxy.tolist()[0]
    print(bbox)

try:
    print(bbox)

    # Extract the values
    x0, y0, width, height = bbox

# Open an image file
image = Image.open(image_path)

# Get its width and height
img_width, img_height = image.size

print("The image dimensions are {}x{}".format(img_width, img_height))

# Calculate the center of the bounding box
x_center = (x0 + width)/ 2
y_center = (y0 + height)/ 2

print("The center of the bounding box is at ({}, {})".format(x_center, y_center))

# Create a draw object
draw = ImageDraw.Draw(image)

# Draw a dot at the center of the bounding box
draw.ellipse((x_center-5, y_center-5, x_center+5, y_center+5), fill='red')

# Save the image
image.save("/Users/kcoolyt/Desktop/Khalid/Research/pythonProject1/example_with_center_dot2.jpg")

#Calculate degree to pixel ratio
DPP = FOV/img_width
print("Degree per pixel ratio is equal to:",DPP)

Angletofire = x_center * DPP

print(Angletofire)