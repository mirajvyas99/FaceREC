import face_recognition as fr
import cv2
import numpy as np
import os

# Step 1: Prepare the dataset
train_path = r"C:\Miraj\UoG\Image Analysis W24\Face Recognition Project\train"
test_path = r"C:\Miraj\UoG\Image Analysis W24\Face Recognition Project\test"

known_names = []
known_name_encodings = []

# Step 2: Train the model
train_images = os.listdir(train_path)
for image_name in train_images:
    image = fr.load_image_file(os.path.join(train_path, image_name))
    encoding = fr.face_encodings(image)[0]
    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(image_name)[0].capitalize())

# Step 3: Test the model on the test dataset
test_image = os.path.join(test_path, "test.jpg")
image = cv2.imread(test_image)
face_locations = fr.face_locations(image)
face_encodings = fr.face_encodings(image, face_locations)
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = fr.compare_faces(known_name_encodings, face_encoding)
    name = ""
    face_distances = fr.face_distance(known_name_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_names[best_match_index]
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Display the result
cv2.imshow("Result", image)
cv2.imwrite(r"C:\Miraj\UoG\Image Analysis W24\Face Recognition Project\output.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
