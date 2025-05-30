import cv2

model = 'car_detect_1000_2000/cascade.xml'
cap = cv2.VideoCapture("images/test_video.avi")


while True:
    ret, image = cap.read()

    if not ret:
        print("All frames done")
        break


    classifier = cv2.CascadeClassifier(model)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    objects = classifier.detectMultiScale(gray,
                                        scaleFactor=1.1, minNeighbors=10,
                                        minSize=(24, 24))
    
    for (x, y, w, h) in objects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2) 

    cv2.imshow("image", image)

    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()