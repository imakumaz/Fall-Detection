document.addEventListener('DOMContentLoaded', function() {
    const videoInput = document.getElementById('videoFileCombined');
    const csvInput = document.getElementById('csvFileCombined');
    const videoStatus = document.getElementById('videoStatus');
    const csvStatus = document.getElementById('csvStatus');
    const detectButton = document.getElementById('combinedDetectButton');
    const outputText = document.getElementById('combinedOutputText');
    const videoConfidence = document.getElementById('videoConfidence');
    const sensorConfidence = document.getElementById('sensorConfidence');
    const combinedConfidence = document.getElementById('combinedConfidence');
    const confidenceScores = document.querySelector('.confidence-scores');
    const videoContainer = document.getElementById('combinedVideoContainer');
    const plotContainer = document.getElementById('combinedPlotContainer');
    const processedVideo = document.getElementById('combinedProcessedVideo');
    const videoSource = document.getElementById('combinedVideoSource');
    const fallPlot = document.getElementById('combinedFallPlot');

    // Update file status when files are selected
    videoInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            videoStatus.textContent = `Selected: ${this.files[0].name}`;
            videoStatus.style.color = 'green';
        } else {
            videoStatus.textContent = 'No video selected';
            videoStatus.style.color = 'initial';
        }
        checkFilesSelected();
    });

    csvInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            csvStatus.textContent = `Selected: ${this.files[0].name}`;
            csvStatus.style.color = 'green';
        } else {
            csvStatus.textContent = 'No CSV selected';
            csvStatus.style.color = 'initial';
        }
        checkFilesSelected();
    });

    // Enable/disable detect button based on file selection
    function checkFilesSelected() {
        if (videoInput.files.length > 0 && csvInput.files.length > 0) {
            detectButton.disabled = false;
        } else {
            detectButton.disabled = true;
        }
    }

    // Initialize button state
    checkFilesSelected();

    // Handle detection button click
    detectButton.addEventListener('click', async function() {
        if (videoInput.files.length === 0 || csvInput.files.length === 0) {
            alert('Please upload both a video file and a CSV file.');
            return;
        }

        // Update UI to show processing state
        detectButton.textContent = 'Processing...';
        detectButton.style.backgroundColor = 'gray';
        detectButton.disabled = true;
        outputText.textContent = 'Processing files...';
        outputText.style.color = 'blue';
        confidenceScores.style.display = 'none';
        videoContainer.style.display = 'none';
        plotContainer.style.display = 'none';

        // Create form data for upload
        const formData = new FormData();
        formData.append('video_file', videoInput.files[0]);
        formData.append('csv_file', csvInput.files[0]);

        try {
            // Send request to server
            const response = await fetch('/process_combined', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }

            // Parse response
            const data = await response.json();
            console.log('Server response:', data);

            // Display results
            if (data.combined_fall_detected) {
                outputText.innerHTML = `🚨 Fall detected! Combined confidence: ${(data.combined_confidence * 100).toFixed(2)}%`;
                outputText.style.color = 'red';
            } else {
                outputText.innerHTML = '✅ No fall detected.';
                outputText.style.color = 'green';
            }

            // Update confidence scores and show the section
            videoConfidence.textContent = `${(data.video_confidence * 100).toFixed(2)}%`;
            sensorConfidence.textContent = `  ${(data.sensor_confidence * 100).toFixed(2)}%`;
            combinedConfidence.textContent = `${(data.combined_confidence * 100).toFixed(2)}%`;
            confidenceScores.style.display = 'block';

            // Show processed video if available
            if (data.video_url) {
                console.log('Loading video from:', data.video_url);
                
                // Reset video element
                processedVideo.pause();
                processedVideo.removeAttribute('src');
                processedVideo.load();
                
                // Set new source with timestamp to prevent caching
                videoSource.src = data.video_url + '?t=' + new Date().getTime();
                
                // Load and show video
                processedVideo.load();
                videoContainer.style.display = 'block';
                
                // Add error listener
                processedVideo.onerror = function() {
                    console.error('Video error code:', processedVideo.error.code);
                    outputText.textContent += ' (Video error: ' + processedVideo.error.code + ')';
                };
            }

            // Show plot if available
            if (data.plot_url) {
                fallPlot.src = data.plot_url + '?t=' + new Date().getTime();
                plotContainer.style.display = 'block';
            }

        } catch (error) {
            console.error('Error:', error);
            outputText.textContent = '❌ Failed to process files: ' + error.message;
            outputText.style.color = 'red';
        } finally {
            // Reset button state after processing
            detectButton.textContent = 'Detect Fall (Combined Approach)';
            detectButton.style.backgroundColor = '';
            detectButton.disabled = false;
        }
    });
    const quickDetectButton = document.getElementById('quickDetectButton');
    
    if (quickDetectButton) {
        quickDetectButton.addEventListener('click', async function() {
            if (videoInput.files.length === 0 || csvInput.files.length === 0) {
                alert('Please upload both a video file and a CSV file.');
                return;
            }

            // Update UI to show processing state
            quickDetectButton.textContent = 'Processing...';
            quickDetectButton.style.backgroundColor = 'gray';
            quickDetectButton.disabled = true;
            outputText.textContent = 'Quick processing using early-stopping algorithm...';
            outputText.style.color = 'blue';
            confidenceScores.style.display = 'none';
            videoContainer.style.display = 'none';
            plotContainer.style.display = 'none';

            // Create form data for upload
            const formData = new FormData();
            formData.append('video_file', videoInput.files[0]);
            formData.append('csv_file', csvInput.files[0]);

            try {
                // Send request to server
                const response = await fetch('/quick_detect', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }

                // Parse response
                const data = await response.json();
                console.log('Server response:', data);

                // Display results with processing stats
                if (data.combined_fall_detected) {
                    outputText.innerHTML = `🚨 Fall detected! Combined confidence: ${(data.combined_confidence * 100).toFixed(2)}%`;
                    if (data.stats && data.stats.early_stopped) {
                        outputText.innerHTML += ` (Early stopped at ${data.stats.processing_percentage}% of video) <br>Total Frames =${data.stats.total_frames}<br> Processed Frames=  ${data.stats.processed_frames}`;
                    }
                    outputText.style.color = 'red';
                } else {
                    outputText.innerHTML = '✅ No fall detected.';
                    if (data.stats) {
                        outputText.innerHTML += ` Processed ${data.stats.processing_percentage}% of video.<br> Total Frames =${data.stats.total_frames} <br> Processed Frames=  ${data.stats.processed_frames}`;
                    }
                    outputText.style.color = 'green';
                }

                // Update confidence scores and show the section
                videoConfidence.textContent = `${(data.video_confidence * 100).toFixed(2)}%`;
                sensorConfidence.textContent = `${(data.sensor_confidence * 100).toFixed(2)}%`;
                combinedConfidence.textContent = `${(data.combined_confidence * 100).toFixed(2)}%`;
                confidenceScores.style.display = 'block';

                // Show processed video if available
                if (data.video_url) {
                    console.log('Loading video from:', data.video_url);
                    
                    // Reset video element
                    processedVideo.pause();
                    processedVideo.removeAttribute('src');
                    processedVideo.load();
                    
                    // Set new source with timestamp to prevent caching
                    videoSource.src = data.video_url + '?t=' + new Date().getTime();
                    
                    // Load and show video
                    processedVideo.load();
                    videoContainer.style.display = 'block';
                    
                    // Add error listener
                    processedVideo.onerror = function() {
                        console.error('Video error code:', processedVideo.error.code);
                        outputText.textContent += ' (Video error: ' + processedVideo.error.code + ')';
                    };
                }

                // Show plot if available
                if (data.plot_url) {
                    fallPlot.src = data.plot_url + '?t=' + new Date().getTime();
                    plotContainer.style.display = 'block';
                }

            } catch (error) {
                console.error('Error:', error);
                outputText.textContent = '❌ Failed to process files: ' + error.message;
                outputText.style.color = 'red';
            } finally {
                // Reset button state after processing
                quickDetectButton.textContent = 'Quick Detect (Early Stopping)';
                quickDetectButton.style.backgroundColor = '';
                quickDetectButton.disabled = false;
            }
        });
    }
});

   