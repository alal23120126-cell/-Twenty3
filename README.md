# -Twenty3
This project proposes an AI-based virtual mouse and keyboard using hand gestures. A camera captures hand movements, while computer vision and gesture recognition convert predefined gestures into commands for cursor movement, clicking, scrolling, typing and keyboard functions, demonstrating practical applications of AI and human computer interaction
import cv2
import mediapipe as mp
import pyautogui
import math
import time

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
SMOOTHING = 5
PINCH_DISTANCE = 40

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

screen_width, screen_height = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

previous_x = 0
previous_y = 0
last_click_time = 0

print("Virtual Mouse Started")
print("Move your index finger to control the mouse.")
print("Pinch thumb + index finger to click.")
print("Press Q to quit.")

while True:
    success, frame = cap.read()

    if not success:
        print("Could not access webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            index_finger = hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]

            thumb = hand_landmarks.landmark[
                mp_hands.HandLandmark.THUMB_TIP
            ]

            index_x = int(index_finger.x * CAMERA_WIDTH)
            index_y = int(index_finger.y * CAMERA_HEIGHT)

            thumb_x = int(thumb.x * CAMERA_WIDTH)
            thumb_y = int(thumb.y * CAMERA_HEIGHT)

            mouse_x = int(index_finger.x * screen_width)
            mouse_y = int(index_finger.y * screen_height)

            current_x = previous_x + (
                mouse_x - previous_x
            ) / SMOOTHING

            current_y = previous_y + (
                mouse_y - previous_y
            ) / SMOOTHING

            pyautogui.moveTo(
                int(current_x),
                int(current_y)
            )

            previous_x = current_x
            previous_y = current_y

            distance = math.sqrt(
                (index_x - thumb_x) ** 2 +
                (index_y - thumb_y) ** 2
            )

            cv2.line(
                frame,
                (index_x, index_y),
                (thumb_x, thumb_y),
                (255, 0, 255),
                3
            )

            current_time = time.time()

            if distance < PINCH_DISTANCE:
                if current_time - last_click_time > 0.8:
                    pyautogui.click()
                    last_click_time = current_time

                    cv2.putText(
                        frame,
                        "CLICK",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3
                    )

            cv2.putText(
                frame,
                f"Pinch distance: {int(distance)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    cv2.putText(
        frame,
        "Virtual Mouse",
        (20, CAMERA_HEIGHT - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Q = Quit",
        (20, CAMERA_HEIGHT - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("AI Virtual Mouse", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

print("Virtual Mouse stopped.")
