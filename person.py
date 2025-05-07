import cv2
import mediapipe as mp
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize YuNet Face Detector
face_detector = cv2.FaceDetectorYN.create(
    "./face_detection_yunet_2023mar.onnx", 
    "", (320, 320), 0.5, 0.3, 5000,
    cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
)

# Initialize MobileNet SSD for Person Detection
person_net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/mobilenet_iter_73000.caffemodel"
)
PERSON_CLASS_ID = 15  # Class ID for "person"

# ROI Configuration
polygon_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
drawing_polygon = False

# Object Monitoring Configuration
box_start = None
box_end = None
box_defined = False
baseline_roi = None
baseline_edges = None
tamper_threshold = 0.3

# ====== IMPROVED PERSON TRACKING VARIABLES ======
person_timers = {}  # {person_id: entry_time}
history = {}  # {person_id: (center_x, center_y, last_seen_time, bbox)}
person_id_counter = 1  # Auto-increments for new IDs
IOU_THRESHOLD = 0.4  # Minimum IoU to consider a match
OVERSTAY_THRESHOLD = 10  # seconds
OVERSTAY_WARNING_COOLDOWN = 30  # seconds
last_overstay_warning = {}

def mouse_callback(event, x, y, flags, param):
    global polygon_points, drawing_polygon, box_start, box_end, box_defined, baseline_roi, baseline_edges
    
    if drawing_polygon and event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
    elif not drawing_polygon:
        if event == cv2.EVENT_LBUTTONDOWN:
            box_start = (x, y)
            baseline_roi = None
            baseline_edges = None
        elif event == cv2.EVENT_LBUTTONUP:
            box_end = (x, y)
            box_defined = True
            x1, y1 = box_start
            x2, y2 = box_end
            object_roi = frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
            if object_roi.size > 0:
                gray_roi = cv2.cvtColor(object_roi, cv2.COLOR_BGR2GRAY)
                baseline_roi = gray_roi.copy()
                baseline_edges = cv2.Canny(gray_roi, 30, 100)

def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def calculate_iou(box1, box2):
    """Computes Intersection-over-Union between two bounding boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection coordinates
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def detect_persons(frame):
    global person_id_counter, history, person_timers
    
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    person_net.setInput(blob)
    detections = person_net.forward()
    
    current_time = time.time()
    current_frame_persons = []
    current_frame_ids = set()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])
        if confidence > 0.5 and class_id == PERSON_CLASS_ID:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            if is_point_in_polygon(center_point, polygon_points):
                bbox = (x1, y1, x2 - x1, y2 - y1)
                
                # Track using IoU matching
                matched_id = None
                max_iou = 0
                for pid, (prev_cx, prev_cy, last_seen, prev_bbox) in history.items():
                    iou = calculate_iou(bbox, prev_bbox)
                    if iou > max_iou and iou > IOU_THRESHOLD and (current_time - last_seen) < 6:
                        matched_id, max_iou = pid, iou
                
                # Assign new ID if no match
                if matched_id is None:
                    matched_id = person_id_counter
                    person_id_counter += 1
                    person_timers[matched_id] = current_time
                
                # Update history
                history[matched_id] = (center_point[0], center_point[1], current_time, bbox)
                current_frame_ids.add(matched_id)
                
                # Draw bounding box (red if overstaying)
                color = (255, 100, 0)  # Original orange color
                time_in_frame = current_time - person_timers[matched_id]
                if time_in_frame > OVERSTAY_THRESHOLD:
                    color = (0, 0, 255)  # Red for overstaying
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Person {matched_id}", (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                current_frame_persons.append((matched_id, bbox))
    
    # Cleanup: Remove stale entries (not seen in >6 seconds)
    to_remove = [pid for pid in history if (current_time - history[pid][2]) > 6]
    for pid in to_remove:
        history.pop(pid, None)
        person_timers.pop(pid, None)
    
    return frame, current_frame_persons

def check_overcrowding(frame, persons):
    if len(persons) > 1:
        cv2.putText(frame, "WARNING: OVERCROWDING DETECTED!", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def check_overstaying(frame):
    current_time = time.time()
    global last_overstay_warning
    
    for person_id in person_timers:
        entry_time = person_timers[person_id]
        duration = current_time - entry_time
        
        if person_id in history:  # Only check if person is currently visible
            _, _, _, bbox = history[person_id]
            x, y, w, h = bbox
            
            if duration > OVERSTAY_THRESHOLD:
                # Only warn once per cooldown period
                if person_id not in last_overstay_warning or \
                   (current_time - last_overstay_warning.get(person_id, 0)) > OVERSTAY_WARNING_COOLDOWN:
                    cv2.putText(frame, f"Person {person_id} Overstaying!", 
                               (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 0, 255), 2)
                    last_overstay_warning[person_id] = current_time

def detect_face_coverage(frame, persons):
    h, w, _ = frame.shape
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(frame)
    faces = faces if faces is not None else []
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    
    covered_faces = 0
    
    for face in faces:
        x, y, w_box, h_box = map(int, face[:4])
        cx, cy = x + w_box // 2, y + h_box // 2
        
        if not is_point_in_polygon((cx, cy), polygon_points):
            continue
            
        landmarks = face[4:14].reshape((5, 2)).astype(np.int32)
        nose = landmarks[2]
        mouth = ((landmarks[3][0] + landmarks[4][0]) // 2, 
                (landmarks[3][1] + landmarks[4][1]) // 2)
        
        # Check for hand occlusion
        hand_occlusion = False
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    hand_pos = (int(lm.x * w), int(lm.y * h))
                    if ((abs(hand_pos[0] - nose[0]) < 25 and abs(hand_pos[1] - nose[1]) < 25) or
                        (abs(hand_pos[0] - mouth[0]) < 25 and abs(hand_pos[1] - mouth[1]) < 25)):
                        hand_occlusion = True
                        break
        
        # Check for object occlusion
        roi = frame[max(y, 0):min(y+h_box, h), max(x, 0):min(x+w_box, w)]
        object_occlusion = False
        if roi.size > 0:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            skin_mask = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
            skin_coverage = np.sum(skin_mask > 0) / skin_mask.size
            object_occlusion = skin_coverage < 0.3
        
        if hand_occlusion or object_occlusion:
            covered_faces += 1
            
    return covered_faces, len(faces)

 
def detect_tampering(frame):
    global baseline_roi, baseline_edges
    
    if not box_defined or baseline_roi is None:
        return frame
    
    x1, y1 = box_start
    x2, y2 = box_end
    current_roi = frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
    
    if current_roi.size == 0:
        cv2.putText(frame, "ROI out of frame!", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame
    
    gray_roi = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
    current_edges = cv2.Canny(gray_roi, 30, 100)
    
    if baseline_roi.shape == gray_roi.shape:
        edge_diff = cv2.absdiff(baseline_edges, current_edges)
        edge_change = np.count_nonzero(edge_diff) / edge_diff.size
        (score, _) = compare_ssim(baseline_roi, gray_roi, full=True)
        
        if edge_change > tamper_threshold or score < 0.8:
            cv2.rectangle(frame, box_start, box_end, (0, 0, 255), 2)
            cv2.putText(frame, "TAMPER DETECTED!", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "ROI size changed - possible tampering!", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return frame

# Main video processing loop
# rtsp_url = "rtsp://admin:ut123456@192.168.1.250:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif"
# cap = cv2.VideoCapture(rtsp_url)
cap = cv2.VideoCapture(0)
cv2.namedWindow("Monitoring System")
cv2.setMouseCallback("Monitoring System", mouse_callback)

if not cap.isOpened():
    print("Error: Couldn't open video stream.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break

        display_frame = frame.copy()

        # ROI drawing mode
        if drawing_polygon:
            for pt in polygon_points:
                cv2.circle(display_frame, pt, 5, (0, 255, 255), -1)
            if len(polygon_points) > 1:
                cv2.polylines(display_frame, [np.array(polygon_points)], isClosed=False, color=(0, 255, 255), thickness=2)
            cv2.putText(display_frame, "Draw ROI Polygon. Press ENTER to Start, R to Reset", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            if len(polygon_points) >= 3:
                cv2.polylines(display_frame, [np.array(polygon_points)], isClosed=True, color=(0, 255, 255), thickness=2)
                display_frame, persons = detect_persons(display_frame)
                display_frame = detect_face_coverage(display_frame, persons)
                check_overcrowding(display_frame, persons)

        if box_defined:
            display_frame = detect_tampering(display_frame)

        # Display person count
        if not drawing_polygon and len(polygon_points) >= 3:
            cv2.putText(display_frame, f"Persons in ROI: {len(persons)}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Monitoring System", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER key
            if len(polygon_points) >= 3:
                polygon_points = np.array(polygon_points, dtype=np.int32)
                drawing_polygon = False
        elif key == ord('r'):  # Reset
            polygon_points = []
            drawing_polygon = True
            box_start = None
            box_end = None
            box_defined = False
            baseline_roi = None
            baseline_edges = None
            history.clear()
            person_timers.clear()
            person_id_counter = 1
            last_overstay_warning.clear()
        elif key == ord('q'):  # Quit
            break

cap.release()
cv2.destroyAllWindows()