import cv2
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()

# Step 1: Load Haar Cascade model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Step 2: Load image
image = cv2.imread("face.jpg")

if image is None:
    raise FileNotFoundError("Image not found. Place face.jpg in the same folder.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5
)

# Step 4: Draw bounding boxes
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Step 5: Show result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Face Detection Result")
plt.show()

