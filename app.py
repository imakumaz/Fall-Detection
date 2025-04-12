from flask import Flask, request, jsonify, render_template, send_from_directory, Response, url_for
import os
import cv2
import cvzone
import math
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from sensor import test_fall_detection, visualize_results
import mimetypes

# Initialize Flask app FIRST
app = Flask(__name__)

# Ensure upload and output directories exist
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SENSOR_UPLOAD_FOLDER = os.path.join(BASE_DIR, "sensor_uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")
COMBINED_UPLOAD_FOLDER = os.path.join(BASE_DIR, "combined_uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(SENSOR_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMBINED_UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SENSOR_UPLOAD_FOLDER"] = SENSOR_UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER
app.config["COMBINED_UPLOAD_FOLDER"] = COMBINED_UPLOAD_FOLDER

# Load YOLO model
model = YOLO("fall_detection2/weights/best.pt")
MODEL_PATH = "sensor_models/cnn_lstm_model.h5"  # Update with actual model path
DATA_PATH = "sensor_models/"

# Define class names manually
classnames = ["fall", "not fall"]

# NOW define routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/combined")
def combined():
    return render_template("combined.html")

@app.route("/processed/<filename>")
def get_processed_video(filename):
    # Ensure the file exists
    video_path = os.path.join(app.config["PROCESSED_FOLDER"], filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404
    
    # Set the correct MIME type based on the file extension
    mimetype = None
    if filename.endswith('.mp4'):
        mimetype = 'video/mp4'
    elif filename.endswith('.avi'):
        mimetype = 'video/x-msvideo'
    else:
        mimetype = 'application/octet-stream'
    
    # Use direct file serving with proper MIME type
    return send_from_directory(
        app.config["PROCESSED_FOLDER"], 
        filename, 
        mimetype=mimetype,
        as_attachment=False
    )

@app.route("/process_video", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded file
    filename = secure_filename(video_file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    video_file.save(filepath)

    # Open video file
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return jsonify({"error": "Failed to open video"}), 500

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define output video file - ensure it has .mp4 extension
    output_filename = "processed_" + os.path.splitext(filename)[0] + ".mp4"
    output_path = os.path.join(app.config["PROCESSED_FOLDER"], output_filename)
    
    # Use H.264 codec for better browser compatibility
    try:
        if os.name == 'nt':  # Windows
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        else:  # Linux/Mac
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        # Fallback to mp4v if H264/avc1 not available
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Process video frames
    fall_detected = False
    max_confidence = 0
    
    # Processing loop
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        results = model(frame)

        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_detect = int(box.cls[0])
                class_detect = classnames[class_detect]
                conf = math.ceil(confidence * 100)

                height = y2 - y1
                width = x2 - x1
                threshold = height - width

                if conf > 50:
                    # Update max confidence if this detection has higher confidence
                    if conf > max_confidence:
                        max_confidence = conf
                        
                    cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                    cvzone.putTextRect(frame, f"{class_detect} {conf}%", [x1 + 8, y1 - 12], thickness=2, scale=2)

                if class_detect == "fall" and threshold < 0:
                    cvzone.putTextRect(frame, "Fall Detected", [x1, y1 - 20], thickness=2, scale=2)
                    fall_detected = True

        # Write processed frame to video
        out.write(frame)

    cap.release()
    out.release()

    # Return URL with proper path
    video_url = f"/processed/{output_filename}"
    
    return jsonify({
        "message": "Processing completed", 
        "fall_detected": fall_detected, 
        "confidence_score": max_confidence / 100,  # Convert to [0,1] range
        "video_url": video_url
    })


@app.route("/detect-fall", methods=["POST"])
def detect_fall():
    if "csv_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["csv_file"]
    filepath = os.path.join(SENSOR_UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Run fall detection
    predictions_prob, predictions, windows = test_fall_detection(
        MODEL_PATH, DATA_PATH, filepath, window_size=200, overlap=0.5, threshold=0.5
    )

    # Identify fall windows
    fall_windows = [i for i, pred in enumerate(predictions) if pred == 1]

    # If at least one fall window is detected, mark fall as detected
    detected_fall = len(fall_windows) > 0

    # Compute confidence score as the average probability of detected fall windows
    confidence_score = (
        sum(predictions_prob[i][0] for i in fall_windows) / len(fall_windows)
        if detected_fall
        else 0.0
    )

    # Visualize results
    plot_path = visualize_results(predictions_prob, predictions, windows, file.filename)

    if plot_path:
        plot_url = f"/resultsplot/{os.path.basename(plot_path)}"
    else:
        plot_url = None 
    
    # Prepare result data
    result = {
        "total_windows": int(len(predictions)),
        "fall_detected": bool(detected_fall),
        "fall_windows": [int(i) for i in fall_windows],
        "probabilities": [float(p) for p in predictions_prob.flatten().tolist()],
        "confidence_score": float(round(confidence_score, 4)) if detected_fall else 0.0,
        "plot_url": plot_url
    }

    return jsonify(result)

@app.route('/resultsplot/<filename>')
def serve_plot(filename):
    return send_from_directory("resultsplot", filename)

@app.route("/process_combined", methods=["POST"])
def process_combined():
    # Check if both files are uploaded
    if "video_file" not in request.files or "csv_file" not in request.files:
        return jsonify({"error": "Both video and CSV files must be uploaded"}), 400

    video_file = request.files["video_file"]
    csv_file = request.files["csv_file"]
    
    if video_file.filename == "" or csv_file.filename == "":
        return jsonify({"error": "Both files must be selected"}), 400

    # Save uploaded files
    video_filename = secure_filename(video_file.filename)
    csv_filename = secure_filename(csv_file.filename)
    
    video_filepath = os.path.join(app.config["COMBINED_UPLOAD_FOLDER"], video_filename)
    csv_filepath = os.path.join(app.config["COMBINED_UPLOAD_FOLDER"], csv_filename)
    
    video_file.save(video_filepath)
    csv_file.save(csv_filepath)

    # Process video using existing logic
    # Open video file
    cap = cv2.VideoCapture(video_filepath)
    if not cap.isOpened():
        return jsonify({"error": "Failed to open video"}), 500

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define output video file
    output_filename = f"combined_processed_{os.path.splitext(video_filename)[0]}.mp4"
    output_path = os.path.join(app.config["PROCESSED_FOLDER"], output_filename)
    
    # Use appropriate codec
    try:
        if os.name == 'nt':  # Windows
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        else:  # Linux/Mac
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        # Fallback
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Process video frames
    video_fall_detected = False
    video_max_confidence = 0
    
    # Processing loop
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        results = model(frame)

        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_detect = int(box.cls[0])
                class_detect = classnames[class_detect]
                conf = math.ceil(confidence * 100)

                height = y2 - y1
                width = x2 - x1
                threshold = height - width

                if conf > 50:
                    # Update max confidence if this detection has higher confidence
                    if conf > video_max_confidence:
                        video_max_confidence = conf
                        
                    cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                    cvzone.putTextRect(frame, f"{class_detect} {conf}%", [x1 + 8, y1 - 12], thickness=2, scale=2)

                if class_detect == "fall" and threshold < 0:
                    cvzone.putTextRect(frame, "Fall Detected", [x1, y1 - 20], thickness=2, scale=2)
                    video_fall_detected = True

        # Write processed frame to video
        out.write(frame)

    cap.release()
    out.release()

    # Process CSV using existing logic
    predictions_prob, predictions, windows = test_fall_detection(
        MODEL_PATH, DATA_PATH, csv_filepath, window_size=200, overlap=0.5, threshold=0.5
    )

    # Identify fall windows
    fall_windows = [i for i, pred in enumerate(predictions) if pred == 1]

    # If at least one fall window is detected, mark fall as detected
    sensor_fall_detected = len(fall_windows) > 0

    # Compute sensor confidence score
    sensor_confidence_score = (
        sum(predictions_prob[i][0] for i in fall_windows) / len(fall_windows)
        if sensor_fall_detected
        else 0.0
    )

    # Visualize sensor results
    plot_path = visualize_results(predictions_prob, predictions, windows, f"combined_{csv_filename}")

    if plot_path:
        plot_url = f"/resultsplot/{os.path.basename(plot_path)}"
    else:
        plot_url = None

    # Normalize video confidence to [0,1] range
    video_confidence = video_max_confidence / 100

    # Calculate combined result using weighted average
    # You can adjust the weights based on which method is more reliable
    video_weight = 0.7
    sensor_weight = 0.3
    
    # Only consider combined fall detected if either method detects a fall
    # combined_fall_detected = video_fall_detected or sensor_fall_detected
    
    # Calculate combined confidence score
    if video_fall_detected and sensor_fall_detected:
        # Both detected fall, use weighted average
        combined_confidence = (video_confidence * video_weight) + (sensor_confidence_score * sensor_weight)
    elif video_fall_detected:
        # Only video detected fall
        combined_confidence = video_confidence * video_weight
    elif sensor_fall_detected:
        # Only sensor detected fall
        combined_confidence = sensor_confidence_score * sensor_weight
    else:
        combined_confidence = 0.0

    
    combined_fall_detected = bool(combined_confidence > 0.4)
    # Return combined results
    video_url = f"/processed/{output_filename}"
    
    return jsonify({
        "message": "Combined processing completed",
        "video_fall_detected": video_fall_detected,
        "sensor_fall_detected": sensor_fall_detected,
        "combined_fall_detected": combined_fall_detected,
        "video_confidence": float(video_confidence),
        "sensor_confidence": float(sensor_confidence_score),
        "combined_confidence": float(combined_confidence),
        "video_url": video_url,
        "plot_url": plot_url
    })

@app.route("/quick_detect", methods=["POST"])
def quick_detect():
    # Check if both files are uploaded
    if "video_file" not in request.files or "csv_file" not in request.files:
        return jsonify({"error": "Both video and CSV files must be uploaded"}), 400

    video_file = request.files["video_file"]
    csv_file = request.files["csv_file"]
    
    if video_file.filename == "" or csv_file.filename == "":
        return jsonify({"error": "Both files must be selected"}), 400

    # Save uploaded files
    video_filename = secure_filename(video_file.filename)
    csv_filename = secure_filename(csv_file.filename)
    
    video_filepath = os.path.join(app.config["COMBINED_UPLOAD_FOLDER"], video_filename)
    csv_filepath = os.path.join(app.config["COMBINED_UPLOAD_FOLDER"], csv_filename)
    
    video_file.save(video_filepath)
    csv_file.save(csv_filepath)

    # Define output video path
    output_filename = f"optimized_processed_{os.path.splitext(video_filename)[0]}.mp4"
    output_path = os.path.join(app.config["PROCESSED_FOLDER"], output_filename)
    video_url = f"/processed/{output_filename}"
    
    # STEP 1: Start with video processing for early detection
    cap = cv2.VideoCapture(video_filepath)
    if not cap.isOpened():
        return jsonify({"error": "Failed to open video"}), 500

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Choose codec
    try:
        if os.name == 'nt':  # Windows
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        else:  # Linux/Mac
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        # Fallback
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize video variables
    video_fall_detected = False
    video_confidence = 0
    processed_frames = 0
    early_stopped = False
    
    # Define early stopping parameters
    confidence_threshold = 0.70  # If confidence exceeds this, we can stop
    min_frames_percentage = 0.15  # Process at least this percentage of video frames
    max_frames_percentage = 0.8   # Process at most this percentage if no confident detection
    
    min_frames = max(int(total_frames * min_frames_percentage), 30)  # At least 30 frames
    max_frames = int(total_frames * max_frames_percentage)
    
    # Process video with potential early stopping
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        processed_frames += 1
        
        # Process current frame
        results = model(frame)
        current_frame_has_fall = False
        current_frame_confidence = 0
        
        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_detect = int(box.cls[0])
                class_detect = classnames[class_detect]
                conf = math.ceil(confidence * 100)
                
                height = y2 - y1
                width = x2 - x1
                threshold = height - width
                
                if conf > 50:
                    # Update confidence if this detection has higher confidence
                    norm_conf = conf/100  # Normalize to [0,1]
                    if norm_conf > current_frame_confidence:
                        current_frame_confidence = norm_conf
                    
                    cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                    cvzone.putTextRect(frame, f"{class_detect} {conf}%", [x1 + 8, y1 - 12], thickness=2, scale=2)
                
                if class_detect == "fall" and threshold < 0 and conf > 50:
                    cvzone.putTextRect(frame, "Fall Detected", [x1, y1 - 20], thickness=2, scale=2)
                    current_frame_has_fall = True
        
        # Update overall video confidence and fall detection status
        if current_frame_has_fall and current_frame_confidence > video_confidence:
            video_confidence = current_frame_confidence
            video_fall_detected = True
        
        # Write processed frame to video
        out.write(frame)
        
        # EARLY STOPPING CONDITIONS:
        # 1. We've processed minimum frames AND
        # 2. We've found a fall with high confidence
        if (processed_frames >= min_frames and 
            video_fall_detected and 
            video_confidence > confidence_threshold):
            early_stopped = True
            break
            
        # Also stop if we've processed enough frames
        if processed_frames >= max_frames:
            break
    
    # Close video handling
    cap.release()
    out.release()
    
    # STEP 2: Now process sensor data (faster operation)
    predictions_prob, predictions, windows = test_fall_detection(
        MODEL_PATH, DATA_PATH, csv_filepath, window_size=200, overlap=0.5, threshold=0.5
    )

    # Calculate sensor results
    fall_windows = [i for i, pred in enumerate(predictions) if pred == 1]
    sensor_fall_detected = len(fall_windows) > 0
    
    sensor_confidence_score = (
        sum(predictions_prob[i][0] for i in fall_windows) / len(fall_windows)
        if sensor_fall_detected
        else 0.0
    )

    # Create plot for sensor data
    plot_path = visualize_results(predictions_prob, predictions, windows, f"optimized_{csv_filename}")
    plot_url = f"/resultsplot/{os.path.basename(plot_path)}" if plot_path else None
    
    # STEP 3: Calculate combined confidence with adaptive weighting
    # If video already found fall with high confidence, we weight it higher
    if video_fall_detected and video_confidence > 0.7:
        video_weight = 0.7
        sensor_weight = 0.3
    else:
        video_weight = 0.6
        sensor_weight = 0.4
    
    # Calculate combined confidence based on what data we have
    if video_fall_detected and sensor_fall_detected:
        # Both detected fall, use weighted average
        combined_confidence = (video_confidence * video_weight) + (sensor_confidence_score * sensor_weight)
    elif video_fall_detected:
        # Only video detected fall
        combined_confidence = video_confidence * video_weight
    elif sensor_fall_detected:
        # Only sensor detected fall
        combined_confidence = sensor_confidence_score * sensor_weight
    else:
        combined_confidence = 0
    
    combined_fall_detected = bool(combined_confidence > 0.4)
    
    # Return results along with processing stats
    return jsonify({
        "message": "Optimized detection completed" + (" (early stopped)" if early_stopped else ""),
        "video_fall_detected": video_fall_detected,
        "sensor_fall_detected": sensor_fall_detected,
        "combined_fall_detected": combined_fall_detected,
        "video_confidence": float(video_confidence),
        "sensor_confidence": float(sensor_confidence_score),
        "combined_confidence": float(combined_confidence),
        "video_url": video_url,
        "plot_url": plot_url,
        "stats": {
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "processing_percentage": round((processed_frames / total_frames) * 100, 1) if total_frames > 0 else 0,
            "early_stopped": early_stopped
        }
    })
if __name__ == "__main__":
    app.run(debug=True)