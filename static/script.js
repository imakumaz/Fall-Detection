document.getElementById("detectButton").addEventListener("click", async () => {
    const videoInput = document.getElementById("videoUpload").files[0];
    const videoOutputText = document.getElementById("videoOutputText");
    const processedVideo = document.getElementById("processedVideo");
    const videoSource = document.getElementById("videoSource");
    const detectButton = document.getElementById("detectButton");
    const videoContainer = document.getElementById("videoContainer");
    
    if (!videoInput) {
        alert("Please upload a video first.");
        return;
    }
    
    // Change button text and color when clicked
    detectButton.textContent = "Processing...";
    detectButton.style.backgroundColor = "gray";
    detectButton.disabled = true;
    
    let formData = new FormData();
    formData.append("video", videoInput);
    
    try {
        const response = await fetch("/process_video", {
            method: "POST",
            body: formData,
        });
        
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log("Server response:", data);
        
        if (data.fall_detected) {
            videoOutputText.textContent = `🚨 Fall detected! Confidence: ${data.confidence_score}`;
            videoOutputText.style.color = "red";
        } else {
            videoOutputText.textContent = "✅ No fall detected.";
            videoOutputText.style.color = "green";
        }
        
        // Show the video player and set the source
        if (data.video_url) {
            console.log("Loading video from:", data.video_url);
            
            // Reset video element
            processedVideo.pause();
            processedVideo.removeAttribute('src');
            processedVideo.load();
            
            // Set new source with timestamp to prevent caching
            videoSource.src = data.video_url + "?t=" + new Date().getTime();
            videoSource.type = "video/mp4";
            
            // Show debugging info
            console.log("Video element:", processedVideo);
            console.log("Source element:", videoSource);
            
            // Load and show video
            processedVideo.load();
            videoContainer.style.display = "block";
            
            // Add error listener
            processedVideo.onerror = function() {
                console.error("Video error code:", processedVideo.error.code);
                videoOutputText.textContent += " (Video error: " + processedVideo.error.code + ")";
            };
            
            // Try to play automatically
            try {
                const playPromise = processedVideo.play();
                if (playPromise !== undefined) {
                    playPromise.catch(error => {
                        console.log("Auto-play prevented:", error);
                        // Auto-play was prevented, this is expected in many browsers
                    });
                }
            } catch (e) {
                console.log("Play error:", e);
            }
        }
        
    } catch (error) {
        console.error("Error:", error);
        videoOutputText.textContent = "❌ Failed to process video: " + error.message;
        videoOutputText.style.color = "red";
    } finally {
        // Reset button state after processing
        detectButton.textContent = "Detect Fall";
        detectButton.style.backgroundColor = ""; // Reset to default color
        detectButton.disabled = false;
    }
});
document.getElementById("detectSensorButton").addEventListener("click", function () {
    const fileInput = document.getElementById("csvUpload");
    const resultText = document.getElementById("csvoutputText");
    const plotImage = document.getElementById("fallPlot");
    const detectSensorButton = document.getElementById("detectSensorButton");
   
    if (fileInput.files.length === 0) {
        resultText.textContent = "Please upload a CSV file first.";
        return;
    }
    detectSensorButton.textContent = "Processing...";
    detectSensorButton.style.backgroundColor = "gray";
    detectSensorButton.disabled = true;
    
    const formData = new FormData();
    formData.append("csv_file", fileInput.files[0]);  // Make sure this matches the key in Flask
   
    fetch("/detect-fall", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.fall_detected) {
            resultText.textContent = `🚨 Fall Detected! Confidence score: ${parseFloat((data.confidence_score * 100).toFixed(2))}`;
            resultText.style.color = "red";
        } else {
            resultText.textContent = "No fall detected.";
            resultText.style.color = "green";
        }
        if (data.plot_url) {
            plotImage.src = data.plot_url;
            plotImage.style.display = "block"; // Show image
        } else {
            plotImage.style.display = "none"; // Hide image if no plot
        }
    })
    .catch(error => {
        console.error("Error:", error);
        resultText.textContent = "Error processing file. Please try again.";
        resultText.style.color = "red";
    })
    .finally(() => {
        // Reset button state after processing
        detectSensorButton.textContent = "Detect Fall";
        detectSensorButton.style.backgroundColor = ""; // Reset to default color
        detectSensorButton.disabled = false;
    });
});
