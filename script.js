document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const uploadBox = document.getElementById('upload-box');
    const predictButton = document.getElementById('predict-button');
    const imagePreview = document.getElementById('image-preview');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultsDiv = document.getElementById('results');
    const predictionText = document.getElementById('prediction-text');
    const confidenceText = document.getElementById('confidence-text');
    const originalImage = document.getElementById('original-image');
    const heatmapImage = document.getElementById('heatmap-image');
    const errorMessage = document.getElementById('error-message');

    // Drag and drop functionality
    uploadBox.addEventListener('click', () => fileInput.click());
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });
    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('dragover');
    });
    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    });

    function handleFile(file) {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');
                imagePreviewContainer.classList.remove('hidden');
                predictButton.disabled = false;
                hideResults();
            };
            reader.readAsDataURL(file);
        } else {
            alert('Please upload a valid image file.');
            fileInput.value = '';
        }
    }

    predictButton.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) {
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        
        showLoading();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Server error occurred.');
            }

            const data = await response.json();
            showResults(data);

        } catch (error) {
            showError(error.message);
            console.error('Prediction failed:', error);
        } finally {
            hideLoading();
        }
    });

    function showLoading() {
        loadingSpinner.classList.remove('hidden');
        predictButton.disabled = true;
        hideResults();
        hideError();
    }

    function hideLoading() {
        loadingSpinner.classList.add('hidden');
        predictButton.disabled = false;
    }

    function showResults(data) {
        predictionText.textContent = `Prediction: ${data.predicted_class.replace(/_/g, ' ')}`;
        confidenceText.textContent = `Confidence: ${data.confidence}`;
        
        originalImage.src = `data:image/png;base64,${data.original_image_base64}`;
        
        if (data.heatmap_image_base64) {
            heatmapImage.src = `data:image/png;base64,${data.heatmap_image_base64}`;
            heatmapImage.parentNode.classList.remove('hidden');
        } else {
            heatmapImage.parentNode.classList.add('hidden');
        }
        
        resultsDiv.classList.remove('hidden');
    }

    function hideResults() {
        resultsDiv.classList.add('hidden');
    }

    function showError(message) {
        errorMessage.textContent = `Error: ${message}`;
        errorMessage.classList.remove('hidden');
    }

    function hideError() {
        errorMessage.classList.add('hidden');
    }
});