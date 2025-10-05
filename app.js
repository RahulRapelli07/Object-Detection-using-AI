// Real-Time Object Detection Application
class ObjectDetectionApp {
    constructor() {
        // Application state
        this.model = null;
        this.videoElement = null;
        this.canvas = null;
        this.ctx = null;
        this.isDetecting = false;
        this.stream = null;
        this.animationFrame = null;
        this.modelLoaded = false;
        this.cameraReady = false;
        
        // Configuration from provided data
        this.cocoClasses = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];
        
        this.colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9", "#F8C471", "#82E0AA", "#F1948A", "#85C1E9", "#D7BDE2", "#A3E4D7", "#FAD7A0", "#D5A6BD", "#AED6F1", "#ABEBC6"];
        
        // Settings
        this.settings = {
            confidenceThreshold: 0.5,
            showConfidence: true,
            detectionEnabled: false,
            maxDetections: 20
        };
        
        // Statistics
        this.stats = {
            totalDetections: 0,
            fps: 0,
            detectionTime: 0,
            activeObjects: 0,
            frameCount: 0,
            lastTime: Date.now()
        };
        
        // Detection data
        this.currentDetections = [];
        this.classCounts = {};
        
        // Initialize the application
        this.init();
    }
    
    async init() {
        try {
            this.setupDOMElements();
            this.setupEventListeners();
            this.renderCocoClasses();
            
            // Load the model first
            await this.loadModel();
            
            // Then initialize camera
            await this.initializeCamera();
            
        } catch (error) {
            console.error('Initialization error:', error);
            this.showError('Failed to initialize the application: ' + error.message);
        }
    }
    
    setupDOMElements() {
        // Get DOM elements
        this.videoElement = document.getElementById('videoElement');
        this.canvas = document.getElementById('detectionCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // UI elements
        this.modelStatus = document.getElementById('modelStatus');
        this.cameraOverlay = document.getElementById('cameraOverlay');
        this.errorMessage = document.getElementById('errorMessage');
        this.errorText = document.getElementById('errorText');
        this.loadingSpinner = document.getElementById('loadingSpinner');
        
        // Controls
        this.toggleButton = document.getElementById('toggleDetection');
        this.clearButton = document.getElementById('clearDetections');
        this.confidenceSlider = document.getElementById('confidenceSlider');
        this.confidenceValue = document.getElementById('confidenceValue');
        this.showConfidenceToggle = document.getElementById('showConfidenceToggle');
        this.enableSoundToggle = document.getElementById('enableSoundToggle');
        this.cameraSelect = document.getElementById('cameraSelect');
        this.retryButton = document.getElementById('retryButton');
        this.classSearch = document.getElementById('classSearch');
        
        // Statistics
        this.totalDetectionsEl = document.getElementById('totalDetections');
        this.fpsCounterEl = document.getElementById('fpsCounter');
        this.detectionTimeEl = document.getElementById('detectionTime');
        this.activeObjectsEl = document.getElementById('activeObjects');
        this.detectedObjectsList = document.getElementById('detectedObjectsList');
        this.cocoClassesList = document.getElementById('cocoClassesList');
    }
    
    setupEventListeners() {
        // Detection controls
        this.toggleButton.addEventListener('click', () => this.toggleDetection());
        this.clearButton.addEventListener('click', () => this.clearDetections());
        this.retryButton.addEventListener('click', () => this.initializeCamera());
        
        // Settings controls
        this.confidenceSlider.addEventListener('input', (e) => {
            this.settings.confidenceThreshold = parseFloat(e.target.value);
            this.confidenceValue.textContent = this.settings.confidenceThreshold.toFixed(1);
        });
        
        this.showConfidenceToggle.addEventListener('change', (e) => {
            this.settings.showConfidence = e.target.checked;
        });
        
        this.cameraSelect.addEventListener('change', (e) => {
            if (e.target.value) {
                this.switchCamera(e.target.value);
            }
        });
        
        // Search functionality
        this.classSearch.addEventListener('input', (e) => {
            this.filterCocoClasses(e.target.value);
        });
        
        // Video element events
        this.videoElement.addEventListener('loadedmetadata', () => {
            this.setupCanvas();
            this.hideOverlay();
            this.cameraReady = true;
            this.updateButtonStates();
        });
        
        this.videoElement.addEventListener('error', () => {
            this.showError('Video stream error occurred');
        });
    }
    
    async loadModel() {
        try {
            this.modelStatus.textContent = 'Loading Model...';
            this.modelStatus.className = 'status status--loading';
            
            // Check if TensorFlow.js is available
            if (typeof tf === 'undefined') {
                throw new Error('TensorFlow.js not loaded');
            }
            
            if (typeof cocoSsd === 'undefined') {
                throw new Error('COCO-SSD model not loaded');
            }
            
            // Load COCO-SSD model with timeout
            const modelPromise = cocoSsd.load();
            const timeoutPromise = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Model loading timeout')), 30000)
            );
            
            this.model = await Promise.race([modelPromise, timeoutPromise]);
            
            this.modelStatus.textContent = 'Model Ready';
            this.modelStatus.className = 'status status--ready';
            this.modelLoaded = true;
            this.updateButtonStates();
            
            console.log('COCO-SSD model loaded successfully');
        } catch (error) {
            console.error('Model loading error:', error);
            this.modelStatus.textContent = 'Model Load Failed';
            this.modelStatus.className = 'status status--error';
            this.showError('Failed to load AI model: ' + error.message);
            throw error;
        }
    }
    
    async initializeCamera() {
        try {
            this.showOverlay('Requesting Camera Access...');
            
            // Check if getUserMedia is available
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Camera API not supported in this browser');
            }
            
            // Get available cameras
            await this.getCameraDevices();
            
            // Request camera access with timeout
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            };
            
            const streamPromise = navigator.mediaDevices.getUserMedia(constraints);
            const timeoutPromise = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Camera access timeout')), 10000)
            );
            
            this.stream = await Promise.race([streamPromise, timeoutPromise]);
            this.videoElement.srcObject = this.stream;
            
            // Update overlay message
            this.showOverlay('Initializing Video...');
            
        } catch (error) {
            console.error('Camera initialization error:', error);
            let errorMessage = 'Unable to access camera. ';
            
            if (error.name === 'NotAllowedError') {
                errorMessage += 'Please grant camera permission and try again.';
            } else if (error.name === 'NotFoundError') {
                errorMessage += 'No camera device found.';
            } else if (error.name === 'NotReadableError') {
                errorMessage += 'Camera is already in use by another application.';
            } else {
                errorMessage += error.message;
            }
            
            this.showError(errorMessage);
        }
    }
    
    async getCameraDevices() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            
            // Populate camera select
            this.cameraSelect.innerHTML = '<option value="">Select Camera</option>';
            videoDevices.forEach((device, index) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.textContent = device.label || `Camera ${index + 1}`;
                this.cameraSelect.appendChild(option);
            });
            
        } catch (error) {
            console.error('Error getting camera devices:', error);
        }
    }
    
    async switchCamera(deviceId) {
        try {
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
            }
            
            const constraints = {
                video: {
                    deviceId: { exact: deviceId },
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                }
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoElement.srcObject = this.stream;
            
        } catch (error) {
            console.error('Camera switch error:', error);
            this.showError('Failed to switch camera: ' + error.message);
        }
    }
    
    setupCanvas() {
        this.canvas.width = this.videoElement.videoWidth;
        this.canvas.height = this.videoElement.videoHeight;
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
    }
    
    updateButtonStates() {
        const canDetect = this.modelLoaded && this.cameraReady;
        this.toggleButton.disabled = !canDetect;
        
        if (!canDetect) {
            this.toggleButton.textContent = 'Initializing...';
            this.toggleButton.className = 'btn btn--secondary btn--full-width';
        } else if (!this.isDetecting) {
            this.toggleButton.textContent = 'Start Detection';
            this.toggleButton.className = 'btn btn--primary btn--full-width';
        }
    }
    
    async toggleDetection() {
        if (!this.model) {
            this.showError('Model not loaded yet');
            return;
        }
        
        if (!this.videoElement.srcObject) {
            this.showError('Camera not initialized');
            return;
        }
        
        this.isDetecting = !this.isDetecting;
        
        if (this.isDetecting) {
            this.toggleButton.textContent = 'Stop Detection';
            this.toggleButton.className = 'btn btn--stop btn--full-width';
            this.startDetection();
        } else {
            this.toggleButton.textContent = 'Start Detection';
            this.toggleButton.className = 'btn btn--primary btn--full-width';
            this.stopDetection();
        }
    }
    
    startDetection() {
        this.stats.lastTime = Date.now();
        this.stats.frameCount = 0;
        this.detectFrame();
    }
    
    stopDetection() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }
    
    async detectFrame() {
        if (!this.isDetecting || !this.model) return;
        
        try {
            const startTime = Date.now();
            
            // Perform detection
            const predictions = await this.model.detect(this.videoElement);
            
            const detectionTime = Date.now() - startTime;
            this.stats.detectionTime = detectionTime;
            
            // Filter predictions by confidence threshold
            const filteredPredictions = predictions.filter(
                pred => pred.score >= this.settings.confidenceThreshold
            );
            
            // Update detections
            this.currentDetections = filteredPredictions.slice(0, this.settings.maxDetections);
            this.updateStatistics();
            this.drawDetections();
            this.updateDetectionsList();
            this.updateClassCounts();
            
            // Calculate FPS
            this.stats.frameCount++;
            const now = Date.now();
            if (now - this.stats.lastTime >= 1000) {
                this.stats.fps = Math.round((this.stats.frameCount * 1000) / (now - this.stats.lastTime));
                this.stats.frameCount = 0;
                this.stats.lastTime = now;
                this.updateStatisticsDisplay();
            }
            
        } catch (error) {
            console.error('Detection error:', error);
            this.showError('Detection failed: ' + error.message);
            this.stopDetection();
        }
        
        // Schedule next frame
        this.animationFrame = requestAnimationFrame(() => this.detectFrame());
    }
    
    drawDetections() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw bounding boxes
        this.currentDetections.forEach((prediction, index) => {
            const [x, y, width, height] = prediction.bbox;
            const confidence = Math.round(prediction.score * 100);
            const label = prediction.class;
            
            // Get color for this detection
            const color = this.colors[index % this.colors.length];
            
            // Draw bounding box
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(x, y, width, height);
            
            // Draw label background
            this.ctx.fillStyle = color;
            const labelText = this.settings.showConfidence ? `${label} (${confidence}%)` : label;
            this.ctx.font = '16px Arial, sans-serif';
            const textMetrics = this.ctx.measureText(labelText);
            const textWidth = textMetrics.width;
            const textHeight = 20;
            
            this.ctx.fillRect(x, y - textHeight - 4, textWidth + 12, textHeight + 4);
            
            // Draw label text
            this.ctx.fillStyle = '#ffffff';
            this.ctx.fillText(labelText, x + 6, y - 8);
        });
    }
    
    updateStatistics() {
        this.stats.totalDetections += this.currentDetections.length;
        this.stats.activeObjects = this.currentDetections.length;
    }
    
    updateStatisticsDisplay() {
        this.totalDetectionsEl.textContent = this.stats.totalDetections;
        this.fpsCounterEl.textContent = this.stats.fps;
        this.detectionTimeEl.textContent = `${this.stats.detectionTime}ms`;
        this.activeObjectsEl.textContent = this.stats.activeObjects;
    }
    
    updateDetectionsList() {
        if (this.currentDetections.length === 0) {
            this.detectedObjectsList.innerHTML = '<p class="no-detections">No objects detected</p>';
            return;
        }
        
        const detectionCounts = {};
        this.currentDetections.forEach(detection => {
            const className = detection.class;
            if (!detectionCounts[className]) {
                detectionCounts[className] = {
                    count: 0,
                    maxConfidence: 0
                };
            }
            detectionCounts[className].count++;
            detectionCounts[className].maxConfidence = Math.max(
                detectionCounts[className].maxConfidence,
                detection.score
            );
        });
        
        let html = '';
        Object.entries(detectionCounts).forEach(([className, data]) => {
            const confidence = Math.round(data.maxConfidence * 100);
            html += `
                <div class="detected-object-item">
                    <span class="object-name">${className} (${data.count})</span>
                    <span class="object-confidence">${confidence}%</span>
                </div>
            `;
        });
        
        this.detectedObjectsList.innerHTML = html;
    }
    
    updateClassCounts() {
        // Reset counts
        this.classCounts = {};
        this.cocoClasses.forEach(className => {
            this.classCounts[className] = 0;
        });
        
        // Count current detections
        this.currentDetections.forEach(detection => {
            if (this.classCounts.hasOwnProperty(detection.class)) {
                this.classCounts[detection.class]++;
            }
        });
        
        // Update display
        this.renderCocoClasses();
    }
    
    renderCocoClasses() {
        const searchTerm = this.classSearch.value.toLowerCase();
        let html = '';
        
        this.cocoClasses
            .filter(className => className.toLowerCase().includes(searchTerm))
            .forEach(className => {
                const count = this.classCounts[className] || 0;
                const countClass = count > 0 ? '' : 'zero';
                
                html += `
                    <div class="coco-class-item">
                        <span class="class-name">${className}</span>
                        <span class="class-count ${countClass}">${count}</span>
                    </div>
                `;
            });
        
        this.cocoClassesList.innerHTML = html;
    }
    
    filterCocoClasses(searchTerm) {
        this.renderCocoClasses();
    }
    
    clearDetections() {
        this.currentDetections = [];
        this.stats.totalDetections = 0;
        this.stats.activeObjects = 0;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.updateStatisticsDisplay();
        this.updateDetectionsList();
        this.updateClassCounts();
    }
    
    showOverlay(message) {
        this.loadingSpinner.querySelector('p').textContent = message;
        this.loadingSpinner.style.display = 'block';
        this.errorMessage.classList.add('hidden');
        this.cameraOverlay.classList.remove('hidden');
    }
    
    hideOverlay() {
        this.cameraOverlay.classList.add('hidden');
    }
    
    showError(message) {
        this.errorText.textContent = message;
        this.loadingSpinner.style.display = 'none';
        this.errorMessage.classList.remove('hidden');
        this.cameraOverlay.classList.remove('hidden');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check for browser compatibility
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        document.body.innerHTML = `
            <div style="text-align: center; padding: 50px; font-family: Arial, sans-serif;">
                <h1>Browser Not Supported</h1>
                <p>This application requires a modern browser with camera support.</p>
                <p>Please use Chrome, Firefox, Safari, or Edge.</p>
                <button onclick="location.reload()" style="margin-top: 20px; padding: 10px 20px; font-size: 16px;">Retry</button>
            </div>
        `;
        return;
    }
    
    // Add fallback for missing libraries
    let missingLibraries = [];
    if (typeof tf === 'undefined') {
        missingLibraries.push('TensorFlow.js');
    }
    if (typeof cocoSsd === 'undefined') {
        missingLibraries.push('COCO-SSD');
    }
    
    if (missingLibraries.length > 0) {
        document.body.innerHTML = `
            <div style="text-align: center; padding: 50px; font-family: Arial, sans-serif;">
                <h1>Loading Error</h1>
                <p>Failed to load required libraries: ${missingLibraries.join(', ')}</p>
                <p>Please check your internet connection and refresh the page.</p>
                <button onclick="location.reload()" style="margin-top: 20px; padding: 10px 20px; font-size: 16px;">Retry</button>
            </div>
        `;
        return;
    }
    
    // Initialize the application
    try {
        new ObjectDetectionApp();
    } catch (error) {
        console.error('Failed to initialize application:', error);
        document.body.innerHTML = `
            <div style="text-align: center; padding: 50px; font-family: Arial, sans-serif;">
                <h1>Initialization Error</h1>
                <p>Failed to start the application: ${error.message}</p>
                <button onclick="location.reload()" style="margin-top: 20px; padding: 10px 20px; font-size: 16px;">Retry</button>
            </div>
        `;
    }
});