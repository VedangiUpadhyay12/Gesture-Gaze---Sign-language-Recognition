
import pickle
import cv2
import mediapipe as mp
import numpy as np


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)  # Set max_num_hands to 2


labels_dict = {0: 'A', 1: 'B', 2: 'C',3: 'D',4: 'E',5: 'F',6: 'G',7: 'H',8: 'I',9: 'J',10: 'K',11: 'L',12: 'M',13: 'N',14: 'O',15: 'P',16: 'Q',17: 'R',18: 'S',19: 'T',20: 'U',21: 'V',22: 'W',23: 'X',24: 'Y',25: 'Z'}


character_accuracies = {char: 0 for char in labels_dict.values()}


def find_camera_index():
    for i in range(10):  
        cap = cv2.VideoCapture(i)
        if cap.isOpened():  
            cap.release()
            return i
    return None  

camera_index = find_camera_index()
if camera_index is None:
    print("Error: No camera detected or camera is inaccessible.")
    exit()


cap = cv2.VideoCapture(camera_index)

while True:
    ret, frame = cap.read()
    
    if not ret:  
        print("Error: Unable to capture frame.")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
                

            if data_aux:
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                

                character_accuracy = character_accuracies[predicted_character]
                character_accuracies[predicted_character] = max(character_accuracy, results.multi_handedness[0].classification[0].score)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 4)  # Light blue bounding box (RGB: 173, 216, 230)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)  # Red text (RGB: 0, 0, 255)
                cv2.putText(frame, f'Accuracy: {character_accuracies[predicted_character]*100:.2f}%', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
