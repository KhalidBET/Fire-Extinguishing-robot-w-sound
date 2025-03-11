from picamera import PiCamera
import cv2
from sklearn.mixture import GaussianMixture
import numpy as np  
import RPi.GPIO as GPIO
import time

# Define paths
photo_path = "/home/raspi/Desktop/+New/image.jpg"
hsv_path = "/home/raspi/Desktop/+New/image_hsv.jpg"
image3rules_path = "/home/raspi/Desktop/+New/image3rules.jpg"
final_path = "/home/raspi/Desktop/+New/final_mask.jpg"# Replace with your desired path
final_pro_path = "/home/raspi/Desktop/+New/final_pro.jpg"
#define variables
fire_existence = 0
fire_direction = None
distance = 0
center_x = 0
center_y = 0

#Define driver pins
# Motor 1
in1 = 26
in2 = 16
in3 = 5
in4 = 6

# setting up
GPIO.setmode( GPIO.BCM )
GPIO.setup( in1, GPIO.OUT )
GPIO.setup( in2, GPIO.OUT )
GPIO.setup( in3, GPIO.OUT )
GPIO.setup( in4, GPIO.OUT )

# initializing
GPIO.output( in1, GPIO.LOW )
GPIO.output( in2, GPIO.LOW )
GPIO.output( in3, GPIO.LOW )
GPIO.output( in4, GPIO.LOW )



def cleanup():
    GPIO.output( in1, GPIO.LOW )
    GPIO.output( in2, GPIO.LOW )
    GPIO.output( in3, GPIO.LOW )
    GPIO.output( in4, GPIO.LOW )
    GPIO.cleanup()

def fire_tracking():
    
    height, width, channels = img.shape
    img_center_x = width // 2
    img_center_y = height // 2
    
    if center_x < frame_center_x:
        motordire = "left"
        move = 1
    elif center_x > frame_center_x:
        motordire = "right"
        move = 1       
    else:
        motordire = "center"
        
    # Fire center
    #if center_y < frame_center_y:
     #   motor2di = "up"
      #  motordi = 1
    #elif center_y == frame_center_y:
     #   motor2di = "center"
    #else:
     #   motor2di = "down"
      #  motordi = 1



# Initialize camera and capture photo
camera = PiCamera()
camera.capture(photo_path)
camera.close()

# Read image in BGR format
image = cv2.imread(photo_path)

# Convert BGR to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Save HSV image
cv2.imwrite(hsv_path, hsv_image)

print(f"Photo captured and HSV image saved to: {hsv_path}")

# Define constants
lower_orange = (0, 30, 80)  # Red color range
upper_orange = (50, 255, 255)
lower_deep_red = (320, 30, 80)  # Red color range
upper_deep_red = (360, 255, 255)
min_fire_pixel_count = 20  # Threshold for connected component size
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Kernel shape for opening and closing

# Load the image
image = cv2.imread(hsv_path)

# Rule 1
mask_orange = cv2.inRange(image, lower_orange, upper_orange)
mask_deep_red = cv2.inRange(image, lower_deep_red, upper_deep_red)
mask_rule1 = cv2.bitwise_or(mask_orange, mask_deep_red)

# Rule 2
lower_rule2 = np.array([15, 50, 50])  # Adjust thresholds as needed
upper_rule2 = np.array([30, 255, 255])
mask_rule2 = cv2.inRange(image, lower_rule2, upper_rule2)

# Rule 3
lower_rule3 = np.array([0, 0, 100])  # Adjust thresholds as needed
upper_rule3 = np.array([180, 30, 255])
mask_rule3 = cv2.inRange(image, lower_rule3, upper_rule3)

# Apply morphological operations to reduce noise (for all masks)
mask_rule1 = cv2.erode(mask_rule1, kernel, iterations=2)
mask_rule1 = cv2.dilate(mask_rule1, kernel, iterations=2)
mask_rule2 = cv2.erode(mask_rule2, kernel, iterations=2)
mask_rule2 = cv2.dilate(mask_rule2, kernel, iterations=2)
mask_rule3 = cv2.erode(mask_rule3, kernel, iterations=2)
mask_rule3 = cv2.dilate(mask_rule3, kernel, iterations=2)

# Count valid pixels for each rule
rule1_count = cv2.countNonZero(mask_rule1)
rule2_count = cv2.countNonZero(mask_rule2)
rule3_count = cv2.countNonZero(mask_rule3)


print("Rule 1 satisfied:", rule1_count > min_fire_pixel_count)
print("Rule 2 satisfied:", rule2_count > min_fire_pixel_count)
print("Rule 3 satisfied:", rule3_count > min_fire_pixel_count)

# Check if all rules are satisfied
all_rules_satisfied = rule1_count > min_fire_pixel_count and rule2_count > min_fire_pixel_count and rule3_count > min_fire_pixel_count

if all_rules_satisfied:
    mask = mask_rule1
else:
    mask = 0
    
# Save the masked image for analysis
cv2.imwrite(image3rules_path, mask)

# Apply GMM analysis only if all rules pass
if all_rules_satisfied:
    mask = mask_rule1 
    # Extract features from masked areas (adapt based on your feature extraction logic)
    features = cv2.findNonZero(mask)  # Replace with your feature extraction
    features = features.reshape(features.shape[0], -1)
    print("1")
    # Define and fit the GMM model
    gmm = GaussianMixture(n_components=2)  # Adjust number of components if needed
    gmm.fit(features)
    print("1")

    # Predict probabilities for each pixel belonging to the fire component
    fire_probabilities = gmm.predict_proba(features)[:, 1]
    print("1")
    print(fire_probabilities.mean())
    # Evaluate fire presence based on probability threshold
    is_fire_present = fire_probabilities.mean() > 0.1
    ##############################################################################################################################
    print("1")
    print(mask.shape, mask.dtype)
    
    # Apply morphological post-processing only if fire detected
    if is_fire_present:
        # Filter connected components based on size
        kernel = np.ones((5, 5), np.uint8)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Update results based on post-processing and save final mask
        image3rules = cv2.imread(image3rules_path)
        final_pro = cv2.bitwise_and(image3rules,image3rules, mask = mask)
        cv2.imwrite(final_path, mask)
        cv2.imwrite(final_pro_path,final_pro)
        print("Morphological post-processing applied and final mask saved.")

        # Set fire existence variable to True
        fire_existence = 1
        
        # Create binary mask
        threshold_value = 200
        ret, binary_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        
        if len(binary_mask.shape) > 2:
            gray_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_mask)
        else:
            # Find connected components with statistics (no need for grayscale conversion)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
        
        # Create contour image (adjust for bounding boxes instead of contours)
        contour_image = np.zeros_like(image)  # Use same shape as original image
        # Draw bounding boxes for each blob
        for label in range(1, num_labels):  # Skip background label
            x, y, w, h = stats[label][:4]  # Extract bounding box coordinates
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle
        center_x = x + w //2
        center_y = y + h //2
        
        # Save images
        cv2.imwrite(binary_mask_path, binary_mask)
        cv2.imwrite(contour_image_path, contour_image)
        print("Image saved")

    # Print overall fire detection conclusion
    print("Fire detected")
else:
    print("Image does not satisfy all fire detection rules.")
    fire_existence = 0

fire_tracking()

if motordire == center:
    GPIO.output( in1, GPIO.HIGH )
    GPIO.output( in2, GPIO.LOW )
    GPIO.output( in3, GPIO.HIGH )
    GPIO.output( in4, GPIO.LOW )
elif motordire == right:
    GPIO.output( in1, GPIO.LOW )
    GPIO.output( in2, GPIO.LOW )
    GPIO.output( in3, GPIO.HIGH )
    GPIO.output( in4, GPIO.LOW )
else:
    GPIO.output( in1, GPIO.HIGH )
    GPIO.output( in2, GPIO.LOW )
    GPIO.output( in3, GPIO.LOW )
    GPIO.output( in4, GPIO.LOW )

# Estimate distance (optional, you can remove this if not needed)
fire_area = w * h
frame_area = image.shape[0] * image.shape[1]
if fire_area >= 0.5 * frame_area:
    distance = 1
    


    

