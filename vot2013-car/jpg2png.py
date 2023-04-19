import cv2

for i in range(1, 375):
    frame = cv2.imread(f"frames/{i:08}.jpg")
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"frames-png-gray/{i}.png", frame_gray)
