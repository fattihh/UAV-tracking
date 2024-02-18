import cv2
import yolo_to_coco
import txtOkuma
import draw_box
import matplotlib.pyplot as plt
import performance_parameter

cap = cv2.VideoCapture("tracking.mp4")  # videoyu aç

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # videonun toplam frame sayısı

# Tracker oluşturma
tracker_mosse = cv2.legacy.TrackerMOSSE_create()
tracker_csrt = cv2.TrackerCSRT_create()
tracker_kcf = cv2.TrackerKCF_create()
tracker_median_flow = cv2.legacy.TrackerMedianFlow_create()
tracker_goturn = cv2.TrackerGOTURN_create()
tracker_tld = cv2.legacy.TrackerTLD_create()
tracker_boosting = cv2.legacy.TrackerBoosting_create()
tracker_mil = cv2.legacy.TrackerMIL_create()

# İlk frame okuma
success, img = cap.read()
cv2.destroyAllWindows()

# Başlangıç bounding box koordinatları
bbox = (817, 274, 89, 32)


# Tracker'ları başlatma
tracker_mosse.init(img, bbox)
tracker_csrt.init(img, bbox)
tracker_kcf.init(img, bbox)
tracker_median_flow.init(img, bbox)
tracker_goturn.init(img, bbox)
tracker_tld.init(img, bbox)
tracker_boosting.init(img, bbox)
tracker_mil.init(img, bbox)

#İoU değerlerini saklamak için liste
iou_list_mosse = []
iou_list_csrt = []
iou_list_kcf = []
iou_list_median_flow = []
iou_list_goturn = []
iou_list_tld = []
iou_list_boosting = []
iou_list_mil = []

center_error_list_mosse = []
center_error_list_csrt = []
center_error_list_kcf = []
center_error_list_medianflow = []
center_error_list_goturn = []
center_error_list_tld = []
center_error_list_boosting = []
center_error_list_mil = []

false_frame = 0
true_frame = 0
frame_count_yolo = 0

txt_dosyalari = txtOkuma.txtOku("test/labels")

while True:
    timer_mosse = cv2.getTickCount()

    success, img = cap.read()

    if success == 0:
        break

    success1, bbox_mosse = tracker_mosse.update(img)    #multiprocessing thread utils dosya yapısı
    success2, bbox_csrt = tracker_csrt.update(img)
    success3, bbox_kcf = tracker_kcf.update(img)
    success4, bbox_median_flow = tracker_median_flow.update(img)
    success5, bbox_goturn = tracker_goturn.update(img)
    success6, bbox_tld = tracker_tld.update(img)
    success7, bbox_boosting = tracker_boosting.update(img)
    success8, bbox_mil = tracker_mil.update(img)

    if success1 and success2 and success3 and success4 and success5 and success6 and success7 and success8:
        true_frame += 1

        # Bounding box çizme ve renklendirme
        x1, y1, w1, h1 = draw_box.drawBox(img, bbox_mosse, color=(0, 255, 0))  # Yeşil bbox
        x2, y2, w2, h2 = draw_box.drawBox(img, bbox_csrt, color=(0, 0, 255))  # Kırmızı bbox
        x3, y3, w3, h3 = draw_box.drawBox(img, bbox_kcf, color=(255, 0, 0))  # Mavi bbox
        x4, y4, w4, h4 = draw_box.drawBox(img, bbox_median_flow, color=(0, 255, 255))  # Cyan bbox
        x5, y5, w5, h5 = draw_box.drawBox(img, bbox_goturn, color=(255, 0, 255))  # Magenta bbox
        x6, y6, w6, h6 = draw_box.drawBox(img, bbox_tld, color=(255, 255, 0))  # Sarı bbox
        x7, y7, w7, h7 = draw_box.drawBox(img, bbox_boosting, color=(0, 0, 0))  # Siyah bbox
        x8, y8, w8, h8 = draw_box.drawBox(img, bbox_mil, color=(0, 165, 255))  # Turuncu bbox

        # IoU hesapla
        dosya_yolu = txt_dosyalari[frame_count_yolo]
        with open(dosya_yolu, 'r', encoding='utf-8') as dosya:
            icerik = dosya.readlines()
            for line in icerik:
                values = line.strip().split()
                w, h = img.shape[1], img.shape[0]
                xmin, ymin, xmax, ymax = yolo_to_coco.yolo_label_to_coco(values[1:], w, h)

                bbox_yolo = (xmin, ymin, xmax, ymax)
                bbox_tracker_mosse = (x1, y1, x1 + w1, y1 + h1)
                bbox_tracker_csrt = (x2, y2, x2 + w2, y2 + h2)
                bbox_tracker_kcf = (x3, y3, x3 + w3, y3 + h3)
                bbox_tracker_median_flow = (x4, y4, x4 + w4, y4 + h4)
                bbox_tracker_goturn = (x5, y5, x5 + w5, y5 + h5)
                bbox_tracker_tld = (x6, y6, x6 + w6, y6 + h6)
                bbox_tracker_boosting = (x7, y7, x7 + w7, y7 + h7)
                bbox_tracker_mil = (x8, y8, x8 + w8, y8 + h8)

                iou_mosse = performance_parameter.calculate_iou(bbox_yolo, bbox_tracker_mosse)
                iou_csrt = performance_parameter.calculate_iou(bbox_yolo, bbox_tracker_csrt)
                iou_kcf = performance_parameter.calculate_iou(bbox_yolo, bbox_tracker_kcf)
                iou_median_flow = performance_parameter.calculate_iou(bbox_yolo, bbox_tracker_median_flow)
                iou_goturn = performance_parameter.calculate_iou(bbox_yolo, bbox_tracker_goturn)
                iou_tld = performance_parameter.calculate_iou(bbox_yolo, bbox_tracker_tld)
                iou_boosting = performance_parameter.calculate_iou(bbox_yolo, bbox_tracker_boosting)
                iou_mil = performance_parameter.calculate_iou(bbox_yolo, bbox_tracker_mil)

                iou_list_mosse.append(iou_mosse)
                iou_list_csrt.append(iou_csrt)
                iou_list_kcf.append(iou_kcf)
                iou_list_median_flow.append(iou_median_flow)
                iou_list_goturn.append(iou_goturn)
                iou_list_tld.append(iou_tld)
                iou_list_boosting.append(iou_boosting)
                iou_list_mil.append(iou_mil)

                center1 = performance_parameter.calculate_center_error(bbox_tracker_mosse, bbox_yolo)
                center2 = performance_parameter.calculate_center_error(bbox_tracker_csrt, bbox_yolo)
                center3 = performance_parameter.calculate_center_error(bbox_tracker_kcf, bbox_yolo)
                center4 = performance_parameter.calculate_center_error(bbox_tracker_median_flow, bbox_yolo)
                center5 = performance_parameter.calculate_center_error(bbox_tracker_goturn, bbox_yolo)
                center6 = performance_parameter.calculate_center_error(bbox_tracker_tld, bbox_yolo)
                center7 = performance_parameter.calculate_center_error(bbox_tracker_boosting, bbox_yolo)
                center8 = performance_parameter.calculate_center_error(bbox_tracker_mil, bbox_yolo)

                center_error_list_mosse.append(center1)
                center_error_list_csrt.append(center2)
                center_error_list_kcf.append(center3)
                center_error_list_medianflow.append(center4)
                center_error_list_goturn.append(center5)
                center_error_list_tld.append(center6)
                center_error_list_boosting.append(center7)
                center_error_list_mil.append(center8)

                success_rate_mosse = performance_parameter.calculate_success_rate(iou_list_mosse)
                success_rate_csrt  = performance_parameter.calculate_success_rate(iou_list_csrt)
                success_rate_kcf = performance_parameter.calculate_success_rate(iou_list_kcf)
                success_rate_median = performance_parameter.calculate_success_rate(iou_list_median_flow)
                success_rate_goturn = performance_parameter.calculate_success_rate(iou_list_goturn)
                success_rate_tld = performance_parameter.calculate_success_rate(iou_list_tld)
                success_rate_boosting = performance_parameter.calculate_success_rate(iou_list_boosting)
                success_rate_mil = performance_parameter.calculate_success_rate(iou_list_mil)

        frame_count_yolo += 1

    else:
        false_frame += 1
        frame_count_yolo += 1

        cv2.putText(img, "Bulunamadi", (75, 75), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), thickness=2)


    cv2.imshow("UAV", img)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break


import matplotlib.pyplot as plt

# Ortalama IoU değerlerini hesaplama
average_iou_mosse = sum(iou_list_mosse) / len(iou_list_mosse) if iou_list_mosse else 0
average_iou_csrt = sum(iou_list_csrt) / len(iou_list_csrt) if iou_list_csrt else 0
average_iou_kcf = sum(iou_list_kcf) / len(iou_list_kcf) if iou_list_kcf else 0
average_iou_median_flow = sum(iou_list_median_flow) / len(iou_list_median_flow) if iou_list_median_flow else 0
average_iou_goturn = sum(iou_list_goturn) / len(iou_list_goturn) if iou_list_goturn else 0
average_iou_tld = sum(iou_list_tld) / len(iou_list_tld) if iou_list_tld else 0
average_iou_boosting = sum(iou_list_boosting) / len(iou_list_boosting) if iou_list_boosting else 0
average_iou_mil = sum(iou_list_mil) / len(iou_list_mil) if iou_list_mil else 0

label_iou = ["MOSSE", "CSRT","KCF" ,"MEDIAN", "TLD", "GOTURN","BOOSTING", "MIL"]
iou_values = [average_iou_mosse, average_iou_csrt,average_iou_kcf,average_iou_median_flow, average_iou_tld,
            average_iou_goturn, average_iou_boosting,average_iou_mil]

# Ortalama merkez hata değerini hesaplama
average_center_mosse = sum(center_error_list_mosse) / len(center_error_list_mosse) if center_error_list_mosse else 0
average_center_csrt = sum(center_error_list_csrt) / len(center_error_list_csrt) if center_error_list_csrt else 0
average_center_kcf = sum(center_error_list_kcf) / len(center_error_list_kcf) if center_error_list_kcf else 0
average_center_medianflow = sum(center_error_list_medianflow) / len(center_error_list_medianflow) if center_error_list_medianflow else 0
average_center_goturn = sum(center_error_list_goturn) / len(center_error_list_goturn) if center_error_list_goturn else 0
average_center_tld = sum(center_error_list_tld) / len(center_error_list_tld) if center_error_list_tld else 0
average_center_boosting = sum(center_error_list_boosting) / len(center_error_list_boosting) if center_error_list_boosting else 0
average_center_mil = sum(center_error_list_mil) / len(center_error_list_mil) if center_error_list_mil else 0

#Ortalama FPS değerleri
average_fps_mosse = 350.693
average_fps_csrt = 36.86
average_fps_goturn = 13.98
average_fps_kcf = 175.804
average_fps_medianflow = 93.397
average_fps_tld = 13.826
average_fps_boosting = 44.6
average_fps_mil = 17.876



label_center = ["MOSSE", "CSRT","KCF" ,"MEDIAN", "TLD", "GOTURN","BOOSTING", "MIL"]
center_values = [average_center_mosse, average_center_csrt,average_center_kcf,average_center_medianflow, average_center_tld,
            average_center_kcf, average_center_boosting, average_center_mil]

labels_success_rate = ["MOSSE", "CSRT", "KCF", "MEDIAN", "TLD","GOTURN","BOOSTING", "MIL"]
succes_rate_values = [success_rate_mosse,success_rate_csrt,success_rate_kcf ,success_rate_median, success_rate_tld,
              success_rate_goturn, success_rate_boosting,success_rate_mil]


labels_precsision = ["MOSSE", "CSRT", "KCF", "MEDIAN", "TLD","GOTURN","BOOSTING", "MIL"]
precision_values = [0.4526,0.671,0.2924,0.3005,0.3394,0.15606,0.4142,0.6252]


labels_fps = ["MOSSE", "CSRT",  "KCF", "MEDIAN", "TLD","GOTURN","BOOSTING", "MIL"]
fps_values = [average_fps_mosse, average_fps_csrt,  average_fps_kcf,
              average_fps_medianflow, average_fps_tld,average_fps_goturn, average_fps_boosting, average_fps_mil]


# İlk pencere: Ortalama IoU Değerleri
plt.figure(1)
plt.bar(label_iou, iou_values, color=['green', 'red','blue', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'royalblue'])
plt.ylabel("Ortalama IoU Değeri")
plt.title("Algoritmaların Ortalama IoU Değerleri")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)

# İkinci pencere: Ortalama Merkez Hata Değerleri
plt.figure(2)
plt.bar(label_center, center_values, color=['green', 'red','blue', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'royalblue'])
plt.ylabel("Ortalama Merkez Hata Değeri")
plt.title("Algoritmaların Ortalama Merkez Hata Değerleri")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)

# Üçüncü pencere: Ortalama Başarı Oranı Değerleri
plt.figure(3)
plt.bar(labels_success_rate, succes_rate_values, color=['green', 'red','blue', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'royalblue'])
plt.ylabel("Ortalama Başarı Oranı Değeri")
plt.title("Algoritmaların Ortalama Başarı Oranı Değerleri")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)


# Dördüncü pencere: Ortalama Precision Değerleri
plt.figure(4)
plt.bar(labels_precsision, precision_values, color=['green', 'red','blue', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'royalblue'])
plt.ylabel("Ortalama Precision Değeri")
plt.title("Algoritmaların Ortalama Precision Değerleri")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)

# Besinci pencere: Ortalama FPS Değerleri
plt.figure(5)
plt.bar(labels_fps, fps_values, color=['green', 'red','blue', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'royalblue'])
plt.ylabel("Ortalama FPS Değeri")
plt.title("Algoritmaların Ortalama FPS Değerleri")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)


# Pencereleri göster
plt.show()



cap.release()
cv2.destroyAllWindows()