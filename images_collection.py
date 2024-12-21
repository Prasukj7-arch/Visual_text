import os
import cv2

# Directory where the data will be saved
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes (alphabets A-Z)
number_of_classes = 26
dataset_size = 100

# Open the webcam
cap = cv2.VideoCapture(0)

# Loop over the alphabets
for j in range(number_of_classes):
    # Create a folder for each alphabet if it doesn't exist
    class_name = chr(65 + j)  # 65 is the ASCII value for 'A'
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {class_name}')

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Collecting data for {class_name}. Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        # Save the collected frame with the counter as the filename
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
