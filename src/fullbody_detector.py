import cv2
from random import randrange

cascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"

print("Loading Haar Cascade from:")
print(cascade_path)

trained_fullbody_data = cv2.CascadeClassifier(cascade_path)

webcam_access = cv2.VideoCapture(0)

while True:
    successful_detected_frames, frames = webcam_access.read()

    if not successful_detected_frames:
        print("Could not read frame from webcam.")
        break

    # Convert frame to grayscale
    greyscale_img = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Detect full bodies
    fullbody_coordinates = trained_fullbody_data.detectMultiScale(
        greyscale_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )

    # Draw rectangles around detected full bodies
    for (x, y, w, h) in fullbody_coordinates:

        # Random rectangle color
        color = (
            randrange(256),
            randrange(256),
            randrange(256)
        )

        cv2.rectangle(
            frames,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

    # Display number of detected people
    cv2.putText(
        frames,
        f"People detected: {len(fullbody_coordinates)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show the webcam
    cv2.imshow(
        "Full Body Detector",
        frames
    )

    # Press Q to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# Clean up
webcam_access.release()
cv2.destroyAllWindows()

print("Done")
