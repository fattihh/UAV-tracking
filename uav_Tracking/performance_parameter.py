import numpy as np
def calculate_center_error(predicted_bbox, true_bbox):
    """
    Calculate the center error between predicted and true bounding boxes.

    Parameters:
    - predicted_bbox (tuple): (x, y, w, h) formatında tahmin edilen sınırlayıcı kutu.
    - true_bbox (tuple): (x, y, w, h) formatında gerçek sınırlayıcı kutu.

    Returns:
    - center_error (float): Merkez hatası.
    """
    # Tahmin edilen ve gerçek sınırlayıcı kutuların merkez koordinatlarını hesapla
    pred_center_x = predicted_bbox[0] + predicted_bbox[2] / 2
    pred_center_y = predicted_bbox[1] + predicted_bbox[3] / 2
    true_center_x = true_bbox[0] + true_bbox[2] / 2
    true_center_y = true_bbox[1] + true_bbox[3] / 2

    # Merkez hatalarını hesapla
    center_error = np.sqrt((pred_center_x - true_center_x)**2 + (pred_center_y - true_center_y)**2)
    return center_error

def calculate_success_rate(iou_values, threshold=0.7):
    successful_predictions = sum(iou >= threshold for iou in iou_values)

    # Toplam kare sayısı
    total_boxes = len(iou_values)

    # Success Rate hesapla
    if total_boxes == 0:
        return 0.0  # Handle division by zero
    else:
        success_rate = successful_predictions / total_boxes
        return success_rate

def calculate_precision(true_positives, false_positives):
    if true_positives + false_positives == 0:
        return 0.0  # Handle division by zero
    else:
        precision = true_positives / (true_positives + false_positives)
        return precision

def calculate_iou(bbox1, bbox2):
    # Intersection area hesapla
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    # Union area hesapla
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    union_area = bbox1_area + bbox2_area - intersection_area

    # IoU hesapla
    if union_area == 0:
        return 0.0  # Handle division by zero
    else:
        iou = intersection_area / union_area
        return iou

def calculate_iou(bbox1, bbox2):
    # Intersection area hesapla
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    # Union area hesapla
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    union_area = bbox1_area + bbox2_area - intersection_area

    # IoU hesapla
    if union_area == 0:
        return 0.0  # Handle division by zero
    else:
        iou = intersection_area / union_area
        return iou







