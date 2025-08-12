// Visual Pollution Detection Frontend JavaScript

class VisualPollutionDetector {
    constructor() {
        this.selectedFile = null;
        this.initializeElements();
        this.setupEventListeners();
        this.updateConfidenceDisplay();
    }

    initializeElements() {
        // Core elements
        this.uploadArea = document.getElementById('uploadArea');
        this.imageInput = document.getElementById('imageInput');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.confidenceSlider = document.getElementById('confidence');
        this.confidenceValue = document.getElementById('confidenceValue');

        // Sections
        this.previewSection = document.getElementById('previewSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.errorSection = document.getElementById('errorSection');

        // Preview elements
        this.imagePreview = document.getElementById('imagePreview');
        this.imageInfo = document.getElementById('imageInfo');

        // Result elements
        this.pollutionLevel = document.getElementById('pollutionLevel');
        this.detectionCount = document.getElementById('detectionCount');
        this.avgConfidence = document.getElementById('avgConfidence');
        this.classCounts = document.getElementById('classCounts');
        this.detectionsList = document.getElementById('detectionsList');

        // Error elements
        this.errorText = document.getElementById('errorText');

        // Button elements
        this.btnText = document.querySelector('.btn-text');
        this.btnLoading = document.querySelector('.btn-loading');
    }

    setupEventListeners() {
        // Upload area events
        this.uploadArea.addEventListener('click', () => this.imageInput.click());
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));

        // File input event
        this.imageInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Confidence slider
        this.confidenceSlider.addEventListener('input', () => this.updateConfidenceDisplay());

        // Analyze button
        this.analyzeBtn.addEventListener('click', () => this.analyzeImage());
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file.');
            return;
        }

        // Validate file size (10MB limit)
        const maxSize = 10 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showError('File size too large. Please select an image under 10MB.');
            return;
        }

        this.selectedFile = file;
        this.previewImage(file);
        this.analyzeBtn.disabled = false;
        this.hideError();
    }

    previewImage(file) {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            this.imagePreview.src = e.target.result;
            this.previewSection.style.display = 'block';
            
            // Display image info
            const sizeKB = (file.size / 1024).toFixed(1);
            this.imageInfo.innerHTML = `
                <strong>File:</strong> ${file.name}<br>
                <strong>Size:</strong> ${sizeKB} KB<br>
                <strong>Type:</strong> ${file.type}
            `;
        };
        
        reader.readAsDataURL(file);
    }

    updateConfidenceDisplay() {
        const value = this.confidenceSlider.value;
        this.confidenceValue.textContent = value;
    }

    async analyzeImage() {
        if (!this.selectedFile) {
            this.showError('Please select an image first.');
            return;
        }

        this.setLoading(true);
        this.hideError();
        this.hideResults();

        try {
            const formData = new FormData();
            formData.append('file', this.selectedFile);

            const confidence = this.confidenceSlider.value;
            const response = await fetch(`/api/predict?confidence=${confidence}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }

            const result = await response.json();
            
            if (result.success) {
                this.displayResults(result);
            } else {
                throw new Error(result.error || 'Prediction failed');
            }

        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(`Analysis failed: ${error.message}`);
        } finally {
            this.setLoading(false);
        }
    }

    displayResults(result) {
        // Update summary cards
        this.pollutionLevel.textContent = result.pollution_level;
        this.pollutionLevel.className = `pollution-level ${result.pollution_level}`;
        
        this.detectionCount.textContent = result.num_detections;
        this.avgConfidence.textContent = (result.avg_confidence * 100).toFixed(1) + '%';

        // Display class counts
        this.displayClassCounts(result.class_counts);

        // Display individual detections
        this.displayDetections(result.detections);

        // Show results section
        this.resultsSection.style.display = 'block';

        // Scroll to results
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    displayClassCounts(classCounts) {
        const totalDetections = Object.values(classCounts).reduce((sum, count) => sum + count, 0);
        
        if (totalDetections === 0) {
            this.classCounts.innerHTML = `
                <h4>Detection Summary</h4>
                <div class="class-item">
                    <span class="class-name">No pollution detected</span>
                    <span class="class-count">0</span>
                </div>
            `;
            return;
        }

        let html = '<h4>Detection Summary</h4>';
        
        for (const [className, count] of Object.entries(classCounts)) {
            if (count > 0) {
                html += `
                    <div class="class-item">
                        <span class="class-name">${className.replace('_', ' ')}</span>
                        <span class="class-count">${count}</span>
                    </div>
                `;
            }
        }

        this.classCounts.innerHTML = html;
    }

    displayDetections(detections) {
        if (detections.length === 0) {
            this.detectionsList.innerHTML = `
                <h4>Individual Detections</h4>
                <div style="text-align: center; padding: 20px; color: #666;">
                    No pollution elements detected in this image.
                </div>
            `;
            return;
        }

        let html = '<h4>Individual Detections</h4>';
        
        detections.forEach((detection, index) => {
            const confidence = (detection.confidence * 100).toFixed(1);
            const bbox = detection.bbox;
            
            html += `
                <div class="detection-item">
                    <div class="detection-header">
                        <span class="detection-class">${detection.class.replace('_', ' ')}</span>
                        <span class="detection-confidence">${confidence}%</span>
                    </div>
                    <div class="detection-description">${detection.description}</div>
                    <div class="detection-bbox">
                        Bounding Box: (${bbox.x1.toFixed(0)}, ${bbox.y1.toFixed(0)}) → 
                        (${bbox.x2.toFixed(0)}, ${bbox.y2.toFixed(0)})
                    </div>
                </div>
            `;
        });

        this.detectionsList.innerHTML = html;
    }

    setLoading(isLoading) {
        this.analyzeBtn.disabled = isLoading;
        
        if (isLoading) {
            this.btnText.style.display = 'none';
            this.btnLoading.style.display = 'inline-flex';
        } else {
            this.btnText.style.display = 'inline';
            this.btnLoading.style.display = 'none';
        }
    }

    showError(message) {
        this.errorText.textContent = message;
        this.errorSection.style.display = 'block';
        this.errorSection.scrollIntoView({ behavior: 'smooth' });
    }

    hideError() {
        this.errorSection.style.display = 'none';
    }

    hideResults() {
        this.resultsSection.style.display = 'none';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new VisualPollutionDetector();
});