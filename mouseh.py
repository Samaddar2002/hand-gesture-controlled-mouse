import cv2
import mediapipe as mp
import pyautogui

cam = cv2.VideoCapture(0)
hand_mesh = mp.solutions.hands.Hands()
draw_h=mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output=hand_mesh.process(rgb_frame)
    landmark_points = output.multi_hand_landmarks
    frame_h, frame_w, _ = frame.shape
    
    if landmark_points:
        landmarks=landmark_points[0].landmark
        
        for id, lmd in enumerate(landmarks[5:9]):
            x = int(lmd.x * frame_w)
            y = int(lmd.y * frame_h)
            cv2.circle(frame, (x,y), 5, (0,255,0))
            
        if id == 3:
            screen_x = int(lmd.x * screen_w)
            screen_y = int(lmd.y * screen_h)
            pyautogui.moveTo(screen_x, screen_y)
            
        left=[landmarks[4]]
        for lmd in left:
            x = int(lmd.x * frame_w)
            y = int(lmd.y * frame_h)
            cv2.circle(frame, (x,y), 5, (0,255,255))
            
        if (left[0].y - landmarks[8].y) < 0.07 :
            pyautogui.click()
            pyautogui.sleep(1)
        
    cv2.imshow('Gesture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()