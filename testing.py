import cv2
import mediapipe as mp
import numpy as np
import pickle
# import pyttsx3
#
# engine=pyttsx3.init()
# engine.setProperty('rate', 150)  # Speed of speech
# engine.setProperty('volume', 1.0) # Volume (0.0 to 1.0)


model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

labels_dict = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: 'A', 6: 'B', 7: 'C', 8: 'D', 9: 'L'}
last_prediction = ''

open("output.txt", "w").close()

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_list = results.multi_hand_landmarks

        for hand_landmarks in hand_list:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())


        for i in range(2):
            if i < len(hand_list):
                hand_landmarks = hand_list[i]
                for lm in hand_landmarks.landmark:
                    x = lm.x
                    y = lm.y
                    data_aux.extend([x, y])
                    x_.append(x)
                    y_.append(y)
            else:
                data_aux.extend([0] * 42)


        if len(data_aux) == 84:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            # text=str(predicted_character)
            # engine.say(text)
            # engine.runAndWait()

            if predicted_character != last_prediction:
                print(predicted_character)
                with open("output.txt", "a") as f:
                    f.write(predicted_character+" ")
                last_prediction = predicted_character



            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Sign Detection', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
