import cv2

cap = cv2.VideoCapture("sahne5.mp4")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#tracker = cv2.legacy.TrackerMOSSE_create()
#tracker = cv2.TrackerCSRT_create()
#tracker = cv2.TrackerKCF_create()
#tracker = cv2.legacy.TrackerMedianFlow_create()
#tracker = cv2.legacy.TrackerBoosting_create()
#tracker = cv2.legacy.TrackerMIL_create()
#tracker = cv2.legacy.TrackerTLD_create()
tracker = cv2.TrackerGOTURN_create()

success, img = cap.read()

# Bounding boxi sabit değerlerle başlatın
#x_sabit, y_sabit, w_sabit, h_sabit = 566, 374, 57, 51  # Örnek değerler

#bboxı kendiniz secin
bbox = cv2.selectROI("selectRoı",img,False)
#bbox = (x_sabit, y_sabit, w_sabit, h_sabit)
print(bbox[0],bbox[1],bbox[2],bbox[3])
tracker.init(img, bbox)

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (0, 255, 0), 3, 1)
    cv2.putText(img, "Takip Ediliyor", (75, 75), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), thickness=2)

fps_list = []  # FPS değerlerini saklamak için bir liste oluşturun

false_frame = 0
true_frame = 0

while True:
    timer = cv2.getTickCount()
    success, img = cap.read()

    if success == 0:
        break

    success, bbox = tracker.update(img)

    if success:
        true_frame += 1
        drawBox(img, bbox)

    else:
        false_frame += 1
        cv2.putText(img, "Bulunamadi", (75, 75), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), thickness=2)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    fps_list.append(fps)  # Her kare için FPS değerini listeye ekleyin

    cv2.putText(img, f"FPS: {fps:.2f}", (75, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), thickness=2)

    cv2.imshow("UAV", img)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

average_fps = sum(fps_list) / len(fps_list) if fps_list else 0  # Ortalama FPS hesapla, liste boşsa 0 olarak ayarla

print(f"Toplam frame sayisi {frame_count}")
print(f'Dogru frame sayisi {true_frame}')
print(f'Yanlıs frame sayisi {false_frame}')
print("Ortalama FPS:", average_fps)

cap.release()
cv2.destroyAllWindows()
