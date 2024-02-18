import cv2

def drawBox(img, bbox, color=(0, 255, 0)):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), color, 3, 1)
    cv2.putText(img, "Takip Ediliyor", (75, 75), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), thickness=2)
    return x, y, w, h
