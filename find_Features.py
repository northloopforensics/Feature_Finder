#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
import base64
import time
import io
from datetime import datetime
import webbrowser
import threading

# Image processing libraries
from PIL import Image
from nudenet import NudeDetector

# HTML report generation
from jinja2 import Template

# GUI components
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit, 
    QSlider, QDoubleSpinBox, QSpinBox, QCheckBox, QGroupBox, QSplitter,
    QTabWidget, QLineEdit, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QThread
from PyQt6.QtGui import QIcon, QPixmap, QFont, QTextCursor

# Define supported image extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif")

class AnalysisWorker(QThread):
    """Worker thread for running image analysis in the background"""
    progress_update = pyqtSignal(int, int)  # current, total
    file_processed = pyqtSignal(str)
    status_update = pyqtSignal(str)
    analysis_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, directory, threshold, max_size):
        super().__init__()
        self.directory = directory
        self.threshold = threshold
        self.max_size = max_size
        self.analyzer = ImageAnalyzer(threshold=threshold, max_size=max_size)
        self.analyzer.progress_callback = self.update_progress
        self.analyzer.file_callback = self.update_file
        self.analyzer.status_callback = self.update_status

    def update_progress(self, current, total):
        self.progress_update.emit(current, total)

    def update_file(self, file_path):
        """Emit signal for file being processed"""
        if file_path and isinstance(file_path, str):
            try:
                # Just get the filename without the path
                filename = os.path.basename(file_path)
                self.file_processed.emit(filename)
            except Exception:  # Catch any exception, not just RecursionError
                self.file_processed.emit("Processing file...")
        else:
            self.file_processed.emit("Processing file...")
        
    def update_status(self, message):
        self.status_update.emit(message)

    def run(self):
        try:
            self.status_update.emit("Initializing analyzer...")
            if not self.analyzer.initialize_detector():
                self.error_occurred.emit("Failed to initialize NudeNet detector.")
                return
                
            self.status_update.emit("Scanning directory for images...")
            image_files = self.analyzer.scan_directory(self.directory)
            
            if not image_files:
                self.status_update.emit("No image files found.")
                self.analysis_complete.emit([])
                return
                
            self.status_update.emit(f"Analyzing {len(image_files)} images...")
            self.analyzer.analyze_images(image_files)
            
            self.status_update.emit("Analysis complete.")
            self.analysis_complete.emit(self.analyzer.results)
        except Exception as e:
            self.error_occurred.emit(f"Error during analysis: {str(e)}")


class ImageAnalyzer:
    def __init__(self, threshold=0.5, max_size=10, model_path=None):
        """Initialize the image analyzer with detection threshold and max file size (in MB)"""
        self.detector = None
        self.threshold = threshold
        self.max_size_mb = max_size
        self.model_path = model_path  # Store custom ONNX model path
        self.results = []
        self.stats = {
            "total_files_scanned": 0,
            "images_found": 0,
            "images_analyzed": 0,
            "detections_made": 0,
            "skipped_files": 0,
            "errors": 0
        }
        # Callbacks for GUI updates
        self.progress_callback = None
        self.file_callback = None
        self.status_callback = None
        
    def log(self, message):
        """Log a message and pass to status callback if available"""
        print(message)
        if self.status_callback:
            self.status_callback(message)
        
    def initialize_detector(self):
        """Initialize the NudeNet detector on demand"""
        self.log("Initializing NudeNet detector...")
        try:
            # Pass the model path if provided
            if self.model_path:
                self.detector = NudeDetector(model_path=self.model_path)
            else:
                self.detector = NudeDetector()

            self.log("Detector initialized successfully")
            return True
        except Exception as e:
            self.log(f"Error initializing detector: {e}")
            return False
    
    def scan_directory(self, directory_path):
        """Scan a directory and its subdirectories for image files"""
        self.log(f"Scanning directory: {directory_path}")
        
        start_time = time.time()
        total_files = 0
        image_files = []
        
        # Walk through directory structure
        for root, _, files in os.walk(directory_path):
            for file in files:
                total_files += 1
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    file_path = os.path.join(root, file)
                    image_files.append(file_path)
        
        self.stats["total_files_scanned"] = total_files
        self.stats["images_found"] = len(image_files)
        
        self.log(f"Found {len(image_files)} image files out of {total_files} total files")
        self.log(f"Scan completed in {time.time() - start_time:.2f} seconds")
        
        return image_files
    
    def analyze_images(self, image_files):
        """Process images with NudeNet detector and collect results"""
        if not image_files:
            self.log("No images to analyze")
            return []
        
        # Initialize detector if needed
        if not self.detector and not self.initialize_detector():
            self.log("Could not initialize detector. Aborting analysis.")
            return []
        
        self.log(f"Analyzing {len(image_files)} images (threshold: {self.threshold})...")
        start_time = time.time()
        
        for i, file_path in enumerate(image_files):
            # Update progress
            if self.progress_callback:
                self.progress_callback(i + 1, len(image_files))
            
            if self.file_callback:
                try:
                    # Just pass the filename instead of full path
                    self.file_callback(os.path.basename(file_path))
                except Exception as e:
                    # If any error occurs during callback, log it but continue
                    print(f"Callback error: {e}")
                
            # Show progress in log
            if i % 10 == 0:
                self.log(f"Processing {i+1}/{len(image_files)}: {os.path.basename(file_path)}")
            
            try:
                # Check file size
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if file_size_mb > self.max_size_mb:
                    self.log(f"Skipping large file ({file_size_mb:.1f} MB): {file_path}")
                    self.stats["skipped_files"] += 1
                    continue
                
                # Get image dimensions
                img = Image.open(file_path)
                # Check if image is in an unsupported mode
                if img.mode == 'P':
                    # Convert to RGB for processing
                    img = img.convert('RGB')
                width, height = img.size
                
                # Create thumbnail for the report
                thumbnail_data = self.create_thumbnail(img)
                
                # Detect content using NudeNet
                detections = self.detector.detect(file_path)
                
                # Only include images with detections above threshold
                relevant_detections = []
                has_detections = False
                
                if isinstance(detections, list):
                    for detection in detections:
                        score = detection.get('score', 0)
                        label = detection.get('class', detection.get('label', 'unknown'))
                        if score >= self.threshold:
                            relevant_detections.append({
                                'label': label,
                                'score': score,
                                'box': detection.get('box', [])
                            })
                            has_detections = True
                
                if has_detections:
                    self.results.append({
                        'file_path': file_path,
                        'thumbnail': thumbnail_data,
                        'dimensions': f"{width}x{height}",
                        'file_size': f"{file_size_mb:.2f} MB",
                        'detections': relevant_detections
                    })
                    self.stats["detections_made"] += 1
                
                self.stats["images_analyzed"] += 1
                
            except Exception as e:
                self.log(f"Error processing {file_path}: {e}")
                self.stats["errors"] += 1
        
        analysis_time = time.time() - start_time
        self.log(f"Analysis completed in {analysis_time:.2f} seconds")
        self.log(f"Found {self.stats['detections_made']} images with detections above threshold {self.threshold}")
        
        # Sort results by highest detection score
        self.results.sort(key=lambda x: max([d['score'] for d in x['detections']]), reverse=True)
        return self.results
    
    def create_thumbnail(self, img, max_size=(150, 150)):
        """Create a thumbnail from an image and return base64 encoded data"""
        # Create a copy of the image and resize
        thumb = img.copy()
        
        # Convert to RGB if image is in palette mode or another incompatible mode
        if thumb.mode in ('P', 'RGBA', 'LA') or thumb.mode != 'RGB':
            thumb = thumb.convert('RGB')
            
        thumb.thumbnail(max_size, Image.LANCZOS)
        
        # Convert to base64 for embedding in HTML
        buffer = io.BytesIO()
        thumb.save(buffer, format="JPEG", quality=70)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_str}"

    def generate_report(self, output_path):
        """Generate HTML report with thumbnails and detection results"""
        if not self.results:
            self.log("No results to report")
            return False
        
        # Get current date and time for the report
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract all unique detection labels
        all_labels = set()
        for result in self.results:
            for detection in result['detections']:
                all_labels.add(detection['label'])
        
        # Sort labels alphabetically
        all_labels = sorted(all_labels)
        
        # HTML template using Jinja2
        template_str = """<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Analysis Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
                background-color: #f8f9fa;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .stats {
                background-color: #e9ecef;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .filters {
                background-color: #e9ecef;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .filter-controls {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-bottom: 15px;
            }
            .filter-button {
                background-color: #0078D7;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                cursor: pointer;
            }
            .filter-button:hover {
                background-color: #1688E0;
            }
            .filter-options {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 10px;
            }
            .filter-option {
                display: inline-flex;
                align-items: center;
                background-color: #fff;
                padding: 5px 10px;
                border-radius: 4px;
                border: 1px solid #ddd;
            }
            .filter-option label {
                margin-left: 5px;
                cursor: pointer;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
                background-color: #fff;
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #343a40;
                color: white;
                position: sticky;
                top: 0;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .thumbnail {
                max-width: 150px;
                max-height: 150px;
                display: block;
                margin: 0 auto;
            }
            .detection-list {
                list-style-type: none;
                padding: 0;
            }
            .detection-item {
                margin-bottom: 5px;
                padding: 5px;
                background-color: #f0f0f0;
                border-radius: 3px;
            }
            .score-high {
                color: #dc3545;
                font-weight: bold;
            }
            .score-medium {
                color: #fd7e14;
                font-weight: bold;
            }
            .score-low {
                color: #28a745;
            }
            .footer {
                margin-top: 20px;
                text-align: center;
                color: #6c757d;
                font-size: 0.9rem;
            }
            #filteredCount {
                font-weight: bold;
                margin-left: 10px;
            }
            .threshold-filter {
                margin: 10px 0;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .threshold-filter label {
                min-width: 100px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Image Analysis Report</h1>
            <p>Report generated on: {{ report_date }}</p>
            
            <div class="stats">
                <h2>Analysis Statistics</h2>
                <ul>
                    <li>Total files scanned: {{ stats.total_files_scanned }}</li>
                    <li>Images found: {{ stats.images_found }}</li>
                    <li>Images analyzed: {{ stats.images_analyzed }}</li>
                    <li>Images with detections: {{ stats.detections_made }}</li>
                    <li>Skipped files: {{ stats.skipped_files }}</li>
                    <li>Errors: {{ stats.errors }}</li>
                    <li>Detection threshold: {{ threshold }}</li>
                </ul>
            </div>
            
            <div class="filters">
                <h2>Filter Results</h2>
                
                <div class="filter-controls">
                    <button class="filter-button" id="selectAll">Select All</button>
                    <button class="filter-button" id="deselectAll">Deselect All</button>
                    <button class="filter-button" id="applyFilters">Apply Filters</button>
                    <span id="filteredCount">Showing all {{ results|length }} images</span>
                </div>
                
                <div class="threshold-filter">
                    <label for="scoreThreshold">Min Score:</label>
                    <input type="range" id="scoreThreshold" min="0" max="100" value="0" step="5">
                    <output for="scoreThreshold" id="thresholdValue">0.00</output>
                </div>
                
                <h3>Categories</h3>
                <div class="filter-options" id="categoryFilters">
                    {% for label in all_labels %}
                    <div class="filter-option">
                        <input type="checkbox" id="filter-{{ label }}" class="category-filter" value="{{ label }}" checked>
                        <label for="filter-{{ label }}">{{ label }}</label>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <h2>Detected Images (<span id="resultsCount">{{ results|length }}</span>)</h2>
            <table id="resultsTable">
                <thead>
                    <tr>
                        <th>Thumbnail</th>
                        <th>File Information</th>
                        <th>Detections</th>
                    </tr>
                </thead>
                <tbody>
                    {% for result in results %}
                    <tr class="result-row" data-labels="{{ result.detections|map(attribute='label')|join(',') }}">
                        <td>
                            <img src="{{ result.thumbnail }}" alt="Thumbnail" class="thumbnail">
                        </td>
                        <td>
                            <p><strong>Path:</strong> <a href="file://{{ result.file_path }}">{{ result.file_path }}</a></p>
                            <p><strong>Dimensions:</strong> {{ result.dimensions }}</p>
                            <p><strong>Size:</strong> {{ result.file_size }}</p>
                        </td>
                        <td>
                            <ul class="detection-list">
                                {% for detection in result.detections %}
                                <li class="detection-item" data-score="{{ detection.score }}" data-label="{{ detection.label }}">
                                    <strong>{{ detection.label }}:</strong> 
                                    <span class="
                                        {% if detection.score > 0.8 %}score-high
                                        {% elif detection.score > 0.6 %}score-medium
                                        {% else %}score-low{% endif %}
                                    ">{{ "%.2f"|format(detection.score) }}</span>
                                </li>
                                {% endfor %}
                            </ul>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            
            <div class="footer">
                <p>Generated with Feature Finder - North Loop Consulting, LLC</p>
            </div>
        </div>

        <script>
        (function() {
            // Elements
            const table = document.getElementById('resultsTable');
            const rows = table.querySelectorAll('tbody tr');
            const selectAllBtn = document.getElementById('selectAll');
            const deselectAllBtn = document.getElementById('deselectAll');
            const applyFiltersBtn = document.getElementById('applyFilters');
            const filteredCount = document.getElementById('filteredCount');
            const resultsCount = document.getElementById('resultsCount');
            const categoryFilters = document.querySelectorAll('.category-filter');
            const scoreThreshold = document.getElementById('scoreThreshold');
            const thresholdValue = document.getElementById('thresholdValue');
            
            // Initialize
            let activeFilters = Array.from(categoryFilters).map(cb => cb.value);
            let minScore = 0;
            
            // Update threshold display
            scoreThreshold.addEventListener('input', function() {
                minScore = this.value / 100;
                thresholdValue.textContent = minScore.toFixed(2);
            });
            
            // Select all categories
            selectAllBtn.addEventListener('click', function() {
                categoryFilters.forEach(cb => cb.checked = true);
            });
            
            // Deselect all categories
            deselectAllBtn.addEventListener('click', function() {
                categoryFilters.forEach(cb => cb.checked = false);
            });
            
            // Apply filters
            applyFiltersBtn.addEventListener('click', function() {
                // Get selected categories
                activeFilters = Array.from(categoryFilters)
                    .filter(cb => cb.checked)
                    .map(cb => cb.value);
                
                filterResults();
            });
            
            // Filter results based on selected categories and score threshold
            function filterResults() {
                let visibleCount = 0;
                
                rows.forEach(row => {
                    const rowLabels = row.dataset.labels.split(',');
                    const detections = row.querySelectorAll('.detection-item');
                    
                    // Check if any detection matches our filters
                    let showRow = false;
                    
                    detections.forEach(detection => {
                        const label = detection.dataset.label;
                        const score = parseFloat(detection.dataset.score);
                        
                        if (activeFilters.includes(label) && score >= minScore) {
                            showRow = true;
                        }
                    });
                    
                    // Show/hide the row
                    row.style.display = showRow ? '' : 'none';
                    
                    if (showRow) visibleCount++;
                });
                
                // Update counters
                filteredCount.textContent = `Showing ${visibleCount} of ${rows.length} images`;
                resultsCount.textContent = visibleCount;
            }
            
            // Initialize with all results visible
            filterResults();
        })();
        </script>
    </body>
    </html>"""
        
        # Render template with results and all unique labels
        template = Template(template_str)
        html_content = template.render(
            results=self.results,
            stats=self.stats,
            report_date=report_date,
            threshold=self.threshold,
            all_labels=all_labels
        )
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.log(f"Report generated at: {output_path}")
        return True

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feature Finder")
        self.setMinimumSize(800, 600)
        
        self.analyzer = None
        self.worker_thread = None
        self.results = []
        self.report_path = os.path.join(os.path.expanduser("~"), "image_analysis_report.html")
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        
        
        # Create tabs
        tabs = QTabWidget()
        
        # Tab 1: Analysis Setup
        setup_tab = QWidget()
        setup_layout = QVBoxLayout(setup_tab)
        
        # Directory selection
        dir_group = QGroupBox("Directory Selection")
        dir_layout = QVBoxLayout(dir_group)
        
        dir_input_layout = QHBoxLayout()
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Select a directory to scan...")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_directory)
        
        dir_input_layout.addWidget(QLabel("Directory:"))
        dir_input_layout.addWidget(self.dir_input)
        dir_input_layout.addWidget(browse_button)
        
        dir_layout.addLayout(dir_input_layout)
        setup_layout.addWidget(dir_group)
        
        # Analysis options
        options_group = QGroupBox("Analysis Options")
        options_layout = QVBoxLayout(options_group)
        
        # Threshold setting
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Detection Threshold:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(1, 10)
        self.threshold_slider.setValue(5)  # Default: 0.5
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(1)
        
        self.threshold_value = QDoubleSpinBox()
        self.threshold_value.setRange(0.1, 1.0)
        self.threshold_value.setSingleStep(0.05)
        self.threshold_value.setValue(0.6)  # Default:
        
        # Connect slider and spinbox
        self.threshold_slider.valueChanged.connect(self.update_threshold_from_slider)
        self.threshold_value.valueChanged.connect(self.update_slider_from_threshold)
        
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value)
        options_layout.addLayout(threshold_layout)
        
        # Max file size setting
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Maximum File Size (MB):"))
        self.max_size = QSpinBox()
        self.max_size.setRange(1, 100)
        self.max_size.setValue(20)
        size_layout.addWidget(self.max_size)
        size_layout.addStretch()
        options_layout.addLayout(size_layout)
        
        # Report path
        report_layout = QHBoxLayout()
        report_layout.addWidget(QLabel("Report Path:"))
        self.report_path_input = QLineEdit()
        self.report_path_input.setText(self.report_path)
        report_browse_button = QPushButton("Browse...")
        report_browse_button.clicked.connect(self.browse_report_path)
        
        report_layout.addWidget(self.report_path_input)
        report_layout.addWidget(report_browse_button)
        options_layout.addLayout(report_layout)
        
        setup_layout.addWidget(options_group)
        
        # Action buttons
        action_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Analysis")
        self.start_button.setMinimumHeight(40)
        self.start_button.clicked.connect(self.start_analysis)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        
        action_layout.addWidget(self.start_button)
        action_layout.addWidget(self.cancel_button)
        
        setup_layout.addLayout(action_layout)
        setup_layout.addStretch()
        
        # Tab 2: Progress and Log
        progress_tab = QWidget()
        progress_layout = QVBoxLayout(progress_tab)
        
        # Progress bar
        progress_group = QGroupBox("Analysis Progress")
        progress_inner_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.status_label = QLabel("Ready")
        self.file_label = QLabel("No file being processed")
        
        progress_inner_layout.addWidget(self.progress_bar)
        progress_inner_layout.addWidget(self.status_label)
        progress_inner_layout.addWidget(self.file_label)
        
        progress_layout.addWidget(progress_group)
        
        # Log viewer
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        log_layout.addWidget(self.log_text)
        progress_layout.addWidget(log_group, stretch=1)
        
        # Tab 3: Results
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)

        self.results_label = QLabel("No analysis results yet. Run an analysis first.")
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        results_actions = QHBoxLayout()
        self.open_report_button = QPushButton("Open Report")
        self.open_report_button.clicked.connect(self.open_report)
        self.open_report_button.setEnabled(False)

        results_actions.addStretch()
        results_actions.addWidget(self.open_report_button)

        results_layout.addWidget(self.results_label)
        results_layout.addLayout(results_actions)
        results_layout.addStretch()

        # Feature Finder information
        info_group = QGroupBox("Feature Finder Information")
        info_layout = QVBoxLayout(info_group)
        info_layout.addWidget(QLabel("Feature Finder v1.0"))
        info_layout.addWidget(QLabel("Developed by: North Loop Consulting"))
        info_layout.addWidget(QLabel("""Feature Finder is a tool for detecting and analyzing images using machine learning.
        It scans directories for images, applies detection algorithms to find human body parts, 
        and generates HTML reports with results."""))
        info_layout.addWidget(QLabel("NudeNet is used for image analysis including its ONNX model support. To select a different model, use the optional ONNX model selection below."))
        info_layout.addWidget(QLabel("You can download models from: https://github.com/notAI-tech/NudeNet"))
        results_layout.addWidget(info_group)
        
        # ONNX model selection (moved here under the Results tab)
        onnx_group = QGroupBox("Optional ONNX Model Selection")
        onnx_layout = QVBoxLayout(onnx_group)

        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(QLabel("ONNX Model Path:"))
        self.model_path_input = QLineEdit()
        model_path_layout.addWidget(self.model_path_input)

        model_browse_button = QPushButton("Browse ONNX Model")
        model_browse_button.clicked.connect(self.browse_model_path)
        model_path_layout.addWidget(model_browse_button)

        onnx_layout.addLayout(model_path_layout)
        results_layout.addWidget(onnx_group)
        
        # Add tabs to tab widget
        tabs.addTab(setup_tab, "Setup")
        tabs.addTab(progress_tab, "Progress")
        tabs.addTab(results_tab, "Results/Settings")
        
        main_layout.addWidget(tabs)
        
        # Status bar at the bottom
        self.statusBar().showMessage("Ready")
        
        # Set the central widget
        self.setCentralWidget(main_widget)
        
   
    def log(self, message):
        """Add message to log viewer"""
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.log_text.ensureCursorVisible()  # Auto-scroll to the cursor
        QApplication.processEvents()

    def browse_model_path(self):
        """Open file dialog to select a custom ONNX model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select ONNX Model", "",
            "ONNX Files (*.onnx);;All Files (*)"
        )
        if file_path:
            self.model_path_input.setText(file_path)
        
    def browse_directory(self):
        """Open dialog to select directory to scan"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory to Scan", "",
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.dir_input.setText(directory)
            
    def browse_report_path(self):
        """Open dialog to select report output location"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report As", "",
            "HTML Files (*.html);;All Files (*)"
        )
        if file_path:
            if not file_path.endswith('.html'):
                file_path += '.html'
            self.report_path_input.setText(file_path)
    
    def update_threshold_from_slider(self, value):
        """Update threshold value when slider is moved"""
        threshold = value / 10.0
        self.threshold_value.setValue(threshold)
        
    def update_slider_from_threshold(self, value):
        """Update slider position when threshold value is changed"""
        slider_value = int(value * 10)
        self.threshold_slider.setValue(slider_value)
    
    def start_analysis(self):
        """Start the analysis process"""
        # Validate inputs
        directory = self.dir_input.text().strip()
        if not directory or not os.path.isdir(directory):
            QMessageBox.warning(self, "Input Error", "Please select a valid directory to scan.")
            return
            
        # Get settings
        threshold = self.threshold_value.value()
        max_size = self.max_size.value()
        self.report_path = self.report_path_input.text().strip()
        
        # Update UI state
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting analysis...")
        self.file_label.setText("Initializing...")
        self.log_text.clear()
        
        # Create and start worker thread
        model_path = self.model_path_input.text().strip()
        self.worker_thread = AnalysisWorker(directory, threshold, max_size)
        # Pass model_path to the analyzer constructor
        self.worker_thread.analyzer = ImageAnalyzer(
            threshold=threshold,
            max_size=max_size,
            model_path=model_path if model_path else None
        )
        self.worker_thread = AnalysisWorker(directory, threshold, max_size)
        self.worker_thread.progress_update.connect(self.update_progress)
        self.worker_thread.file_processed.connect(self.update_file)
        self.worker_thread.status_update.connect(self.update_status)
        self.worker_thread.analysis_complete.connect(self.analysis_finished)
        self.worker_thread.error_occurred.connect(self.handle_error)
        self.worker_thread.start()
        
        # Update status bar
        self.statusBar().showMessage("Analysis running...")
        
    def cancel_analysis(self):
        """Cancel the running analysis"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.terminate()
            self.log("Analysis cancelled by user")
            self.status_label.setText("Analysis cancelled")
            self.cancel_button.setEnabled(False)
            self.start_button.setEnabled(True)
            self.statusBar().showMessage("Analysis cancelled")
    
    def update_progress(self, current, total):
        """Update progress bar"""
        percent = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        
    def update_file(self, file_name):
        """Update currently processed file label"""
        # The file_name is already just the basename or a safe string
        self.file_label.setText(f"Processing: {file_name}")
        
    def update_status(self, message):
        """Update status label and log"""
        self.status_label.setText(message)
        self.log(message)
        
    def analysis_finished(self, results):
        """Handle the completion of analysis"""
        self.results = results
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        
        # Generate report
        if results:
            self.log(f"Analysis complete. Found {len(results)} images with detections.")
            self.results_label.setText(f"Analysis found {len(results)} images with detections.")
            
            # Generate the HTML report
            self.log("Generating HTML report...")
            try:
                self.worker_thread.analyzer.generate_report(self.report_path)
                self.open_report_button.setEnabled(True)
                self.log(f"Report generated at: {self.report_path}")
                
                # Offer to open report
                reply = QMessageBox.question(
                    self, 
                    "Analysis Complete",
                    f"Analysis complete.\nFound {len(results)} images with detections.\n\nOpen the report now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.open_report()
            except Exception as e:
                self.log(f"Error generating report: {str(e)}")
                QMessageBox.warning(self, "Report Error", f"Could not generate report: {str(e)}")
        else:
            self.log("Analysis complete. No detection results found.")
            self.results_label.setText("No detection results found.")
            self.open_report_button.setEnabled(False)
            QMessageBox.information(self, "Analysis Complete", "Analysis complete.\nNo detection results found.")
        
        self.statusBar().showMessage("Analysis complete")
    
    def handle_error(self, error_message):
        """Handle errors from the worker thread"""
        self.log(f"ERROR: {error_message}")
        self.status_label.setText("Error")
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_message}")
        self.statusBar().showMessage("Error")
        
    def open_report(self):
        """Open the generated HTML report in the default browser"""
        if os.path.exists(self.report_path):
            webbrowser.open(f"file://{os.path.abspath(self.report_path)}")
            self.log(f"Opening report in browser: {self.report_path}")
        else:
            QMessageBox.warning(self, "File Not Found", "Report file does not exist.")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for a more modern look
    
    # Set application-wide stylesheet for a more attractive UI
    app.setStyleSheet("""
        QGroupBox {
            font-weight: bold;
            border: 1px solid #bbb;
            border-radius: 4px;
            margin-top: 0.5em;
            padding-top: 0.8em;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 3px;
        }
        QPushButton {
            background-color: #0078D7;
            color: white;
            border: none;
            padding: 5px 15px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #1688E0;
        }
        QPushButton:pressed {
            background-color: #006CC1;
        }
        QPushButton:disabled {
            background-color: #888;
        }
        QSlider::groove:horizontal {
            border: 1px solid #bbb;
            height: 10px;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background-color: #0078D7;
            border: 1px solid #777;
            width: 18px;
            margin: -2px 0;
            border-radius: 9px;
        }
    """)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
