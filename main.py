import cv2
import time
import numpy as np
import os

def main():
    # Ensure separate evidence subfolders exist for each trigger type
    evidence_folders = ["snapshots/lingering", "snapshots/frequency", "snapshots/loitering"]
    for folder in evidence_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 1. Initialize Video Capture (0 opens your live webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # 2. Initialize Background Subtractor for Motion Detection (Topic E: Motion & Object Tracking)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # State dictionaries for tracking objects across frames
    tracked_objects = {}     
    next_object_id = 1
    
    # Independent snapshot flags for each rule to prevent overwriting
    lingering_snapshot_saved = False
    frequency_snapshot_saved = False
    loitering_snapshot_saved = False

    print("SmartTrace Security System Active. Press 'q' to exit.")

    # Create window and set it to Fullscreen mode
    window_name = "SmartTrace - Intelligent Security Feed"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or failed to grab frame.")
            break

        current_time = time.time()

        # 3. Frame Preprocessing (Topic A: Image Operations)
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 4. Apply Background Subtractor with a slow learning rate (0.001) 
        # This prevents MOG2 from absorbing you into the background when you stand still!
        fg_mask = bg_subtractor.apply(blurred, learningRate=0.001)
        _, thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up noise blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Dilation to merge fragmented body parts into one object
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        cleaned = cv2.dilate(cleaned, dilate_kernel, iterations=2)

        # 5. Extract Contours (Topic D: Image Analysis & Transformation)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        system_status = "Observing..."
        status_color = (0, 255, 0) # Green for normal

        current_frame_centers = []
        contour_boxes = []

        for contour in contours:
            if cv2.contourArea(contour) < 1000: 
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = int(x + w / 2), int(y + h / 2)
            current_frame_centers.append((cx, cy))
            contour_boxes.append((x, y, w, h))

        # --- CROSS-FRAME OBJECT TRACKING & LOCATION-BASED LOITERING LOGIC ---
        updated_tracked_objects = {}
        
        for cx, cy in current_frame_centers:
            matched_id = None
            min_distance = 100 

            for obj_id, data in tracked_objects.items():
                prev_cx, prev_cy = data["center"]
                distance = np.linalg.norm(np.array([cx, cy]) - np.array([prev_cx, prev_cy]))
                
                if distance < min_distance:
                    matched_id = obj_id
                    min_distance = distance

            if matched_id is not None:
                time_delta = current_time - tracked_objects[matched_id]["last_seen"]
                appearances = tracked_objects[matched_id]["appearances"]
                if tracked_objects[matched_id]["was_missing"]:
                    appearances += 1
                
                # LOCATION CHECK: Measure distance from the object's ORIGINAL starting position
                start_cx, start_cy = tracked_objects[matched_id]["start_center"]
                total_displacement_from_start = np.linalg.norm(np.array([cx, cy]) - np.array([start_cx, start_cy]))
                
                current_stationary_time = tracked_objects[matched_id]["stationary_time"]
                
                if total_displacement_from_start < 120.0:
                    current_stationary_time += time_delta
                else:
                    current_stationary_time = max(0.0, current_stationary_time - 0.5)

                updated_tracked_objects[matched_id] = {
                    "center": (cx, cy),
                    "start_center": (start_cx, start_cy),
                    "start_time": tracked_objects[matched_id]["start_time"],
                    "appearances": appearances,
                    "stationary_time": current_stationary_time,
                    "last_seen": current_time,
                    "was_missing": False
                }
            else:
                new_id = f"obj_{next_object_id}"
                next_object_id += 1
                
                updated_tracked_objects[new_id] = {
                    "center": (cx, cy),
                    "start_center": (cx, cy),
                    "start_time": current_time,
                    "appearances": 1,
                    "stationary_time": 0.0,
                    "last_seen": current_time,
                    "was_missing": False
                }

        for obj_id, data in tracked_objects.items():
            if obj_id not in updated_tracked_objects:
                if current_time - data["last_seen"] < 2.0: 
                    data["was_missing"] = True
                    updated_tracked_objects[obj_id] = data

        tracked_objects = updated_tracked_objects

        # --- 6. BEHAVIORAL FILTERING RULES EVALUATION ---
        for idx, (x, y, w, h) in enumerate(contour_boxes):
            cx, cy = current_frame_centers[idx]
            
            active_id = None
            for obj_id, data in tracked_objects.items():
                if np.linalg.norm(np.array([cx, cy]) - np.array(data["center"])) < 75:
                    active_id = obj_id
                    break
            
            if not active_id:
                continue

            data = tracked_objects[active_id]
            duration = current_time - data["start_time"]
            appearances = data["appearances"]
            stationary_time = data["stationary_time"]

            # --- PREPARE TIMESTAMP STRING FOR ON-SCREEN & SNAPSHOT EVIDENCE ---
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

            # --- EVALUATE MULTI-TIER TRIGGER RULES & SAVING SNAPSHOTS WITH EMBEDDED INFO ---
            if stationary_time >= 5.0:
                system_status = f"LOITERING DETECTED ({int(stationary_time)}s): TRIGGERED!"
                status_color = (0, 0, 255) # Red
                if not loitering_snapshot_saved:
                    evidence_frame = frame.copy()
                    cv2.putText(evidence_frame, f"EVIDENCE: LOITERING ({timestamp_str})", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    filename = os.path.join("snapshots", "loitering", f"loitering_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(filename, evidence_frame)
                    print(f"[ALERT] Loitering rule met! Full-screen snapshot saved to {filename}")
                    loitering_snapshot_saved = True

            elif duration >= 20.0:
                system_status = "LINGERING DETECTED: TRIGGERED!"
                status_color = (0, 0, 255) # Red
                if not lingering_snapshot_saved:
                    evidence_frame = frame.copy()
                    cv2.putText(evidence_frame, f"EVIDENCE: LINGERING (20s) ({timestamp_str})", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    filename = os.path.join("snapshots", "lingering", f"lingering_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(filename, evidence_frame)
                    print(f"[ALERT] Lingering rule met (20s)! Full-screen snapshot saved to {filename}")
                    lingering_snapshot_saved = True

            elif appearances > 3:
                system_status = f"RECURRENT MOTION ({appearances}x): TRIGGERED!"
                status_color = (0, 0, 255) # Red
                if not frequency_snapshot_saved:
                    evidence_frame = frame.copy()
                    cv2.putText(evidence_frame, f"EVIDENCE: FREQUENCY ({appearances}x) ({timestamp_str})", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    filename = os.path.join("snapshots", "frequency", f"frequency_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(filename, evidence_frame)
                    print(f"[ALERT] Frequency rule met ({appearances} appearances)! Full-screen snapshot saved to {filename}")
                    frequency_snapshot_saved = True
            else:
                if duration > 3.0 or appearances > 1:
                    system_status = f"Tracking ({active_id}): AnchorStopTime: {int(stationary_time)}s | Appears: {appearances}x"
                    status_color = (0, 165, 255) # Orange

            # 7. Visualization Overlays (Topic D)
            cv2.rectangle(frame, (x, y), (x + w, y + h), status_color, 2)
            cv2.putText(frame, f"ID: {active_id} (AnchorTime: {int(stationary_time)}s)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        # Draw Global Status and Live Timestamp on Live Feed Window
        cv2.putText(frame, f"Status: {system_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        current_timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_timestamp_str, (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show the processed output frame in fullscreen
        cv2.imshow(window_name, frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()