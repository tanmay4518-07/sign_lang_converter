import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import pyttsx3

# Load the model
model = load_model('sign_model.h5')

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Define classes
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
           'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
           'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space']

# Buffers
sentence = ''
prev_letter = ''
letter_buffer = deque(maxlen=15)

# Webcam setup
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # ROI
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI
    roi_resized = cv2.resize(roi, (64, 64))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = np.expand_dims(roi_normalized, axis=0)

    # Prediction
    pred = model.predict(roi_reshaped)[0]
    class_id = np.argmax(pred)
    confidence = pred[class_id]
    letter = classes[class_id]

    if confidence > 0.95:
        letter_buffer.append(letter)

        if letter_buffer.count(letter) > 10 and letter != prev_letter:
            prev_letter = letter

            if letter == 'space':
                sentence += ' '
            elif letter == 'del':
                sentence = sentence[:-1]
            elif letter == 'nothing':
                pass
            else:
                sentence += letter

    # Display info
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Letter: {letter} ({confidence:.2f})", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, "Sentence: " + sentence, (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Sign to English with TTS", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ''  # Clear sentence
    elif key == ord('t'):
        print("Speaking:", sentence)
        engine.say(sentence)
        engine.runAndWait()

cap.release()
cv2.destroyAllWindows()
