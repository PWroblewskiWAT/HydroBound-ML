#!/usr/bin/env python
# coding: utf-8

"""
HydroBound-ML Assistant
=======================
Version: 1.0.7
Year: 2026

Author:
-------
Patryk Wróblewski
PL: Katedra Rozpoznania Obrazowego, Wydział Inżynierii Lądowej i Geodezji, 
    Wojskowa Akademia Techniczna im. Jarosława Dąbrowskiego, 
    ul. gen. Sylwestra Kaliskiego 2, 00-908 Warszawa.
EN: Department of Imagery Intelligence, Faculty of Civil Engineering and Geodesy, 
    Military University of Technology, 
    2 gen. Sylwestra Kaliskiego St., 00-908 Warsaw, Poland.

Publication Reference:
----------------------
[Insert Title of the Article / DOI Link here upon publication]

Description:
------------
An advanced hybrid machine learning and Object-Based Image Analysis (OBIA) tool 
designed for automated water body delineation and high-precision boundary extraction 
from aerial and UAV-based multispectral imagery.

While fundamentally developed to resolve spatial ambiguity and optimize water classification 
within complex Airborne Lidar Bathymetry (ALB) processing pipelines, the algorithm's 
architecture is highly versatile. It is actively applicable across a broad spectrum of 
Geographic Information Systems (GIS) tasks, including the creation, updating, and 
harmonization of high-resolution topographic databases (e.g., BDOT10k, BDOO), floodplain 
mapping, and general hydrographic vectorization.

This application fuses pixel-level semantic classification (Random Forest operating 
on a 14D spatial-spectral feature space) with geometric regularization via SLIC 
superpixels to ensure topologically sound and physically accurate water-land boundaries. 
It features automated index generation (NDWI, NDVI, NGRDI), dynamic Level of Detail (LoD) 
parameter coupling, and interactive vector topology editing (Boolean operations).
"""

import os
import sys
import json
import numpy as np
import rasterio
import rasterio.features
import rasterio.mask
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import geopandas as gpd
import joblib
import pyproj
import gc
import math
import re
from shapely.geometry import shape, box, Polygon, MultiPolygon, Point
import shapely.ops
from rasterio.windows import from_bounds
import scipy.ndimage as ndimage
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, jaccard_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.cluster import DBSCAN
from skimage.segmentation import slic
import time
import datetime
import fiona

# Configure Fiona to support KML drivers for AOI parsing
try:
    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    fiona.drvsupport.supported_drivers['kml'] = 'rw'
    fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
except Exception:
    pass


def fix_proj():
    """
    Resolves PROJ_LIB environment variable path issues to ensure correct
    coordinate reference system (CRS) transformations in standalone environments.
    """
    p_paths = [os.path.join(sys.prefix, 'Library', 'share', 'proj'), os.path.join(sys.prefix, 'share', 'proj')]
    for p in p_paths:
        if os.path.exists(p):
            os.environ['PROJ_LIB'] = p
            pyproj.datadir.set_data_dir(p)
            return True
    return False


fix_proj()

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QPushButton, QWidget, QFileDialog, QLabel, QMessageBox,
                             QProgressDialog, QRadioButton, QButtonGroup, QCheckBox, QFrame,
                             QSpinBox, QComboBox, QGroupBox, QFormLayout, QDoubleSpinBox, QScrollArea,
                             QInputDialog, QDialog, QDialogButtonBox, QLineEdit)
from PyQt6.QtCore import Qt, QTimer, QCoreApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection


class BandSelectionDialog(QDialog):
    def __init__(self, rgb_count, nir_count=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Spectral Bands")
        self.setModal(True)
        self.setMinimumWidth(350)

        self.needs_nir = nir_count is not None

        layout = QVBoxLayout(self)
        info_label = QLabel("Please map the correct spectral bands from your input orthomosaics:")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        form_layout = QFormLayout()

        self.cb_red = QComboBox()
        self.cb_green = QComboBox()
        self.cb_blue = QComboBox()
        for i in range(1, rgb_count + 1):
            self.cb_red.addItem(f"Band {i}", i)
            self.cb_green.addItem(f"Band {i}", i)
            self.cb_blue.addItem(f"Band {i}", i)

        if rgb_count >= 1: self.cb_red.setCurrentIndex(0)
        if rgb_count >= 2: self.cb_green.setCurrentIndex(1)
        if rgb_count >= 3: self.cb_blue.setCurrentIndex(2)

        form_layout.addRow("Red Band (from RGB ortho):", self.cb_red)
        form_layout.addRow("Green Band (from RGB ortho):", self.cb_green)
        form_layout.addRow("Blue Band (from RGB ortho):", self.cb_blue)

        if self.needs_nir:
            self.cb_nir = QComboBox()
            for i in range(1, nir_count + 1):
                self.cb_nir.addItem(f"Band {i}", i)
            self.cb_nir.setCurrentIndex(0)
            form_layout.addRow("NIR Band (from NIR ortho):", self.cb_nir)

        layout.addLayout(form_layout)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_mapping(self):
        mapping = {
            'red': self.cb_red.currentData(),
            'green': self.cb_green.currentData(),
            'blue': self.cb_blue.currentData()
        }
        if self.needs_nir:
            mapping['nir'] = self.cb_nir.currentData()
        return mapping


class HydroBound_ML_App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HydroBound-ML Assistant v.1.0.7")

        self.base_dir = ""
        self.input_dir = ""
        self.output_dir = ""
        self.model_dir = ""
        self.project_metadata = None
        self.selected_kml_path = ""

        self.data_srcs = {}
        self.layer_norms = {}
        self.session_clicks = []
        self.samples_X, self.samples_y, self.sample_coords = [], [], []

        self.current_layer = "ndwi"
        self.full_extent = [0, 1, 0, 1]
        self.current_model_version = 0

        self.vector_mask = None
        self.vector_lines_cache = []
        self.aoi_lines_cache = []

        self.img_artist = None
        self.vector_artist = None
        self.aoi_artist = None
        self._is_updating = False

        self._pan_active = False
        self._pan_start_x = None
        self._pan_start_y = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None

        self._measure_points = []
        self._measure_artists = []

        self.edit_mode = None  
        self.edit_drawing_points = []
        self.edit_drawing_artists = []

        cmap_blue = LinearSegmentedColormap.from_list("w_blue", ["white", "darkblue"])
        cmap_blue.set_bad(color='white', alpha=0.0)
        cmap_green = LinearSegmentedColormap.from_list("w_green", ["white", "darkgreen"])
        cmap_green.set_bad(color='white', alpha=0.0)
        cmap_red = LinearSegmentedColormap.from_list("w_red", ["white", "darkred"])
        cmap_red.set_bad(color='white', alpha=0.0)
        cmap_dtm = LinearSegmentedColormap.from_list("local_topo", ["#78B856", "#E7D97B", "#B67A41", "#EBEBEB"])
        cmap_dtm.set_bad(color='white', alpha=0.0)

        self.cmaps = {
            "ndwi": cmap_blue, "ndvi": cmap_green, "ngrdi": cmap_red,
            "ortho_rgb": None, "ortho_nir": "gray", "dtm": cmap_dtm
        }

        self.render_timer = QTimer()
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.fetch_high_res_raster)

        self._setup_ui()
        self.showMaximized()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ---------------- LEFT PANEL ----------------
        left_widget = QWidget()
        sidebar = QVBoxLayout(left_widget)
        sidebar.setContentsMargins(10, 10, 10, 10)
        left_widget.setFixedWidth(360)

        self.btns = []
        labels = ["1. Load Project", "2. Predict (OTSU / ML Inference)", "3. Train Model (Update)", "4. Export GeoJSON"]
        for i, text in enumerate(labels):
            b = QPushButton(text)
            b.setMinimumHeight(45)
            self.btns.append(b)
            sidebar.addWidget(b)

        self.btns[0].clicked.connect(self.load_project)
        self.btns[1].clicked.connect(self.predict_mask)
        self.btns[2].clicked.connect(self.train_model)
        self.btns[3].clicked.connect(self.export_geojson)
        
        sidebar.addSpacing(10)
        self.btn_load_poly = QPushButton("Load Existing Polygon (.geojson)")
        self.btn_load_poly.setMinimumHeight(35)
        self.btn_load_poly.clicked.connect(self.load_existing_polygon)
        sidebar.addWidget(self.btn_load_poly)

        sidebar.addSpacing(10)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        sidebar.addWidget(separator)
        sidebar.addSpacing(5)

        self.btn_measure = QPushButton("📏 Measure Distance")
        self.btn_measure.setMinimumHeight(35)
        self.btn_measure.setCheckable(True)
        self.btn_measure.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
            }
            QPushButton:checked {
                background-color: #f0e68c;
                border: 1px solid #daa520;
                font-weight: bold;
                color: black;
            }
        """)
        self.btn_measure.clicked.connect(self.toggle_measure)
        sidebar.addWidget(self.btn_measure)

        sidebar.addSpacing(15)
        self.sample_label = QLabel("Pending ML Clicks -> Water: 0 | Non-Water: 0")
        self.sample_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.sample_label.setWordWrap(True)
        sidebar.addWidget(self.sample_label)

        self.version_label = QLabel("Model Version: None")
        self.version_label.setStyleSheet("color: blue; font-style: italic; font-weight: bold;")
        sidebar.addWidget(self.version_label)
        sidebar.addStretch()

        # ---------------- CENTER PANEL ----------------
        self.map_frame = QFrame()
        self.map_layout = QVBoxLayout(self.map_frame)
        self.fig = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#EAEAEA')

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.map_layout.addWidget(self.toolbar)
        self.map_layout.addWidget(self.canvas)
        self.ax.set_autoscale_on(False)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('axes_enter_event', lambda event: self.canvas.setFocus())
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.ax.callbacks.connect('xlim_changed', self.on_limits_changed)
        self.ax.callbacks.connect('ylim_changed', self.on_limits_changed)

        # ---------------- RIGHT PANEL ----------------
        right_widget = QWidget()
        right_widget.setMinimumWidth(380)
        right_widget.setMaximumWidth(450)
        right_main_layout = QVBoxLayout(right_widget)
        right_main_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        layer_panel = QVBoxLayout(scroll_content)
        layer_panel.setContentsMargins(10, 10, 10, 10)

        layer_panel.addWidget(QLabel("<b>Base Layer</b>"))
        self.layer_group = QButtonGroup(self)
        self.layer_buttons = {}
        layers = ["ndwi", "ndvi", "ngrdi", "ortho_rgb", "ortho_nir", "dtm"]

        for i, name in enumerate(layers):
            rb = QRadioButton(name.upper())
            rb.setEnabled(False)
            self.layer_group.addButton(rb, i)
            self.layer_buttons[name] = rb
            layer_panel.addWidget(rb)

        self.layer_group.idClicked.connect(self.change_layer)

        layer_panel.addSpacing(10)
        self.check_aoi = QCheckBox("Show AOI Mask (KML)")
        self.check_aoi.setStyleSheet("font-weight: bold; color: #DAA520;")
        self.check_aoi.stateChanged.connect(self.toggle_aoi_visibility)
        layer_panel.addWidget(self.check_aoi)

        self.check_mask = QCheckBox("Show Prediction Mask")
        self.check_mask.setStyleSheet("font-weight: bold; color: magenta;")
        self.check_mask.stateChanged.connect(self.toggle_vector_visibility)
        layer_panel.addWidget(self.check_mask)

        # --- STEP 2 ---
        self.pred_group = QGroupBox("Prediction & Post-Processing (Step 2)")
        self.pred_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        pred_layout = QFormLayout(self.pred_group)

        self.spin_prob_thresh = QDoubleSpinBox()
        self.spin_prob_thresh.setRange(0.10, 0.95)
        self.spin_prob_thresh.setValue(0.55)
        self.spin_prob_thresh.setSingleStep(0.05)
        self.spin_noise_filter = QDoubleSpinBox()
        self.spin_noise_filter.setRange(0.0, 100.0)
        self.spin_noise_filter.setValue(0.50)
        self.spin_noise_filter.setSingleStep(0.1)

        self.check_vector_snap = QCheckBox("Enable OBIA Superpixels (SLIC)")
        self.check_vector_snap.setChecked(True)
        self.check_vector_snap.setStyleSheet("color: darkblue;")

        self.spin_slic_area = QDoubleSpinBox()
        self.spin_slic_area.setRange(0.1, 50.0)
        self.spin_slic_area.setSingleStep(0.5)
        self.spin_slic_area.setSuffix(" m²")

        self.spin_slic_compactness = QDoubleSpinBox()
        self.spin_slic_compactness.setRange(0.1, 100.0)
        self.spin_slic_compactness.setValue(15.0)
        self.spin_slic_compactness.setSingleStep(1.0)

        self.spin_obia_ratio = QDoubleSpinBox()
        self.spin_obia_ratio.setRange(0.01, 0.99)
        self.spin_obia_ratio.setValue(0.20)
        self.spin_obia_ratio.setSingleStep(0.05)

        self.spin_smooth_radius = QDoubleSpinBox()
        self.spin_smooth_radius.setRange(0.0, 10.0)
        self.spin_smooth_radius.setSingleStep(0.5)
        self.spin_smooth_radius.setSuffix(" m")

        pred_layout.addRow("Decision Threshold:", self.spin_prob_thresh)
        pred_layout.addRow("Min Area Filter (%):", self.spin_noise_filter)
        pred_layout.addRow("", self.check_vector_snap)
        pred_layout.addRow("Superpixel Size:", self.spin_slic_area)
        pred_layout.addRow("SLIC Compactness:", self.spin_slic_compactness)
        pred_layout.addRow("OBIA Edge Exp. Ratio:", self.spin_obia_ratio)
        pred_layout.addRow("Vector Smooth Radius:", self.spin_smooth_radius)

        # --- STEP 3 ---
        self.train_group = QGroupBox("Model Training Parameters (Step 3)")
        self.train_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        train_layout = QFormLayout(self.train_group)

        self.spin_scale_1 = QDoubleSpinBox()
        self.spin_scale_1.setRange(0.1, 50.0)
        self.spin_scale_1.setValue(1.00)
        self.spin_scale_1.setSingleStep(0.50)
        self.spin_scale_2 = QDoubleSpinBox()
        self.spin_scale_2.setRange(0.2, 100.0)
        self.spin_scale_2.setValue(3.00)
        self.spin_scale_2.setSingleStep(1.0)
        self.spin_scale_3 = QDoubleSpinBox()
        self.spin_scale_3.setRange(0.3, 300.0)
        self.spin_scale_3.setValue(5.00)
        self.spin_scale_3.setSingleStep(1.0)

        self.btn_unlock = QPushButton("Unlock & Reset Model")
        self.btn_unlock.setStyleSheet("color: red; font-size: 10px; font-weight: normal;")
        self.btn_unlock.clicked.connect(self.unlock_scales)
        self.btn_unlock.setEnabled(False)

        self.spin_otsu_water_min = QDoubleSpinBox()
        self.spin_otsu_water_min.setRange(-1.0, 1.0)
        self.spin_otsu_water_min.setValue(-1.00)
        self.spin_otsu_water_min.setSingleStep(0.05)
        self.spin_otsu_land_max = QDoubleSpinBox()
        self.spin_otsu_land_max.setRange(-1.0, 1.0)
        self.spin_otsu_land_max.setValue(1.00)
        self.spin_otsu_land_max.setSingleStep(0.05)

        self.spin_otsu_base = QSpinBox()
        self.spin_otsu_base.setRange(0, 100000)
        self.spin_otsu_base.setValue(500)
        self.spin_otsu_base.setSingleStep(100)

        self.spin_weight = QDoubleSpinBox()
        self.spin_weight.setRange(1.0, 100000.0)
        self.spin_weight.setValue(2.00)

        self.spin_clones = QSpinBox()
        self.spin_clones.setRange(1, 10000)
        self.spin_clones.setValue(20)

        self.spin_noise_aug = QDoubleSpinBox()
        self.spin_noise_aug.setDecimals(3)
        self.spin_noise_aug.setRange(0.0, 0.200)
        self.spin_noise_aug.setValue(0.005)
        self.spin_noise_aug.setSingleStep(0.005)

        self.spin_max_depth = QSpinBox()
        self.spin_max_depth.setRange(0, 100)
        self.spin_max_depth.setValue(0)
        self.spin_max_depth.setSpecialValueText("Unlimited")
        self.spin_min_leaf = QSpinBox()
        self.spin_min_leaf.setRange(1, 100)
        self.spin_min_leaf.setValue(2)

        self.spin_jitter = QDoubleSpinBox()
        self.spin_jitter.setRange(0.0, 10.0)
        self.spin_jitter.setValue(1.00)
        self.spin_jitter.setSingleStep(0.5)
        self.spin_jitter.setSuffix(" m")
        self.spin_jitter_points = QSpinBox()
        self.spin_jitter_points.setRange(0, 50)
        self.spin_jitter_points.setValue(8)

        train_layout.addRow("Scale 1 (Micro):", self.spin_scale_1)
        train_layout.addRow("Scale 2 (Mid):", self.spin_scale_2)
        train_layout.addRow("Scale 3 (Macro):", self.spin_scale_3)
        train_layout.addRow("", self.btn_unlock)
        train_layout.addRow("Otsu Water Min NDWI:", self.spin_otsu_water_min)
        train_layout.addRow("Otsu Land Max NDWI:", self.spin_otsu_land_max)
        train_layout.addRow("Otsu Base Limit:", self.spin_otsu_base)
        train_layout.addRow("Sample Weight:", self.spin_weight)
        train_layout.addRow("Oversample (Clones):", self.spin_clones)
        train_layout.addRow("Clone Feature Noise:", self.spin_noise_aug)
        train_layout.addRow("Max Depth (Gen.):", self.spin_max_depth)
        train_layout.addRow("Min Leaf (Gen.):", self.spin_min_leaf)
        train_layout.addRow("Jitter Radius (m):", self.spin_jitter)
        train_layout.addRow("Jitter Samples:", self.spin_jitter_points)

        # --- STEP 3.5 ---
        self.edit_group = QGroupBox("Vector Editing (Step 3.5)")
        self.edit_group.setStyleSheet("QGroupBox { font-weight: bold; color: darkorange; }")
        edit_layout = QVBoxLayout(self.edit_group)

        self.btn_edit_del = QPushButton("Delete Polygon Part (Click)")
        self.btn_edit_fill = QPushButton("Fill Inner Hole (Click)")
        self.btn_edit_add = QPushButton("Draw Bridge / Add (Left Click, Right to Close)")
        self.btn_edit_cut = QPushButton("Draw Cut / Remove (Left Click, Right to Close)")

        self.edit_btns = [self.btn_edit_del, self.btn_edit_fill, self.btn_edit_add, self.btn_edit_cut]
        for btn in self.edit_btns:
            btn.setCheckable(True)
            btn.clicked.connect(self.toggle_edit_mode)
            edit_layout.addWidget(btn)

        # --- STEP 4 ---
        self.grid_group = QGroupBox("Export Settings (Step 4)")
        self.grid_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        grid_layout = QFormLayout(self.grid_group)

        self.line_epsg = QLineEdit("EPSG:4979")

        self.combo_rfc = QComboBox()
        self.combo_rfc.addItems(["Yes", "No"])
        self.combo_rfc.setCurrentText("Yes")

        self.spin_coord_precision = QSpinBox()
        self.spin_coord_precision.setRange(5, 17)
        self.spin_coord_precision.setValue(15)

        self.check_export_grid = QCheckBox("Enable Grid Export (Requires DTM)")
        self.check_export_grid.setChecked(False)
        self.check_export_grid.stateChanged.connect(self.toggle_grid_settings)

        self.spin_grid_size = QSpinBox()
        self.spin_grid_size.setRange(10, 10000)
        self.spin_grid_size.setValue(500)
        self.spin_grid_size.setSuffix(" m")
        self.spin_grid_size.setEnabled(False)

        self.combo_z_method = QComboBox()
        self.combo_z_method.addItems(["mean", "min", "max"])
        self.combo_z_method.setEnabled(False)

        grid_layout.addRow("Target CRS (EPSG):", self.line_epsg)
        grid_layout.addRow("RFC7946 Format:", self.combo_rfc)
        grid_layout.addRow("Coordinate Precision:", self.spin_coord_precision)
        grid_layout.addRow("DTM Z-Method:", self.combo_z_method) 
        grid_layout.addRow(self.check_export_grid)
        grid_layout.addRow("Grid Size:", self.spin_grid_size)
        

        self.spin_scale_1.valueChanged.connect(lambda v: self.spin_scale_2.setMinimum(v + 0.1))
        self.spin_scale_2.valueChanged.connect(lambda v: self.spin_scale_3.setMinimum(v + 0.1))
        self.spin_scale_1.valueChanged.connect(self.auto_update_postprocessing)

        self.auto_update_postprocessing(self.spin_scale_1.value())

        layer_panel.addSpacing(10)
        layer_panel.addWidget(self.pred_group)
        layer_panel.addSpacing(10)
        layer_panel.addWidget(self.train_group)
        layer_panel.addSpacing(10)
        layer_panel.addWidget(self.edit_group)
        layer_panel.addSpacing(10)
        layer_panel.addWidget(self.grid_group)
        layer_panel.addStretch()
        
        scroll_area.setWidget(scroll_content)
        right_main_layout.addWidget(scroll_area)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.map_frame, 1)
        main_layout.addWidget(right_widget)

    def load_existing_polygon(self):
        if not self.data_srcs:
            QMessageBox.warning(self, "Warning", "Please load a project (Step 1) before loading a polygon, so the coordinate system can be established.")
            return

        if not self.output_dir or not os.path.exists(self.output_dir):
            start_dir = self.base_dir if self.base_dir else ""
        else:
            start_dir = self.output_dir

        file_path, _ = QFileDialog.getOpenFileName(self, "Select Existing Water Boundary Polygon", start_dir, "GeoJSON Files (*.geojson)")
        
        if not file_path:
            return

        if "grid" in os.path.basename(file_path).lower():
            reply = QMessageBox.question(self, "Grid Polygon Detected", 
                                         "You are attempting to load a gridded polygon.\n\n"
                                         "This is highly discouraged because editing or re-exporting it with grid settings will cause overlapping grid intersections and severe performance issues.\n\n"
                                         "It is strongly recommended to load the base (non-grid) polygon. Do you still want to proceed?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

        p = QProgressDialog("Loading Polygon...", "Cancel", 0, 100, self)
        p.setWindowModality(Qt.WindowModality.ApplicationModal)
        p.show()
        QCoreApplication.processEvents()

        try:
            p.setValue(30)
            gdf = gpd.read_file(file_path)
            p.setValue(70)

            gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            
            if gdf.empty:
                QMessageBox.warning(self, "Warning", "Loaded file contains no valid polygons.")
                return

            target_crs = self.get_layer_src(self.current_layer).crs
            if gdf.crs and target_crs and gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)

            self.vector_mask = gdf
            self._cache_vector_lines(self.vector_mask)
            self.check_mask.setChecked(True)
            self.toggle_vector_visibility()
            
            self.edit_mode = None
            for b in self.edit_btns: b.setChecked(False)
            self.btn_measure.setChecked(False)
            self.clear_edit_drawing()
            self.clear_measurements()
            
            QMessageBox.information(self, "Success", f"Successfully loaded polygon with {len(self.vector_mask)} features.\nYou can now proceed to Step 3.5 or Step 4.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load polygon: {e}")
        finally:
            p.setValue(100)
            p.close()

    def toggle_measure(self):
        if self.btn_measure.isChecked():
            self.clear_measurements()
            self.edit_mode = None
            for b in self.edit_btns: b.setChecked(False)
            self.clear_edit_drawing()
        else:
            self.clear_measurements()

    def toggle_edit_mode(self):
        sender = self.sender()
        if not sender.isChecked():
            self.edit_mode = None
            self.clear_edit_drawing()
            return

        for b in self.edit_btns:
            if b != sender: b.setChecked(False)
        self.btn_measure.setChecked(False)
        self.toggle_measure()  
        self.clear_edit_drawing()

        if sender == self.btn_edit_del:
            self.edit_mode = 'del_poly'
        elif sender == self.btn_edit_fill:
            self.edit_mode = 'fill_hole'
        elif sender == self.btn_edit_add:
            self.edit_mode = 'add_poly'
        elif sender == self.btn_edit_cut:
            self.edit_mode = 'cut_poly'

    def clear_edit_drawing(self):
        self.edit_drawing_points = []
        for artist in self.edit_drawing_artists:
            try:
                artist.remove()
            except:
                pass
        self.edit_drawing_artists = []
        self.canvas.draw_idle()

    def clear_measurements(self):
        self._measure_points = []
        for artist in self._measure_artists:
            try:
                artist.remove()
            except:
                pass
        self._measure_artists = []
        self.canvas.draw_idle()

    def auto_update_postprocessing(self, micro_scale_val):
        self.spin_smooth_radius.setValue(micro_scale_val)
        suggested_area = max(0.5, (micro_scale_val ** 2) * 0.7)
        self.spin_slic_area.setValue(round(suggested_area, 2))

    def toggle_grid_settings(self):
        is_checked = self.check_export_grid.isChecked()
        self.spin_grid_size.setEnabled(is_checked)

    def find_latest_model_version(self):
        if not os.path.exists(self.model_dir):
            return 0
        files = os.listdir(self.model_dir)
        versions = []
        for f in files:
            match_model = re.search(r"HydroBound-ML_v(\d+)\.joblib", f)
            if match_model:
                versions.append(int(match_model.group(1)))
            
            match_samples = re.search(r"HydroBound-ML_v(\d+)_manual_samples\.joblib", f)
            if match_samples:
                versions.append(int(match_samples.group(1)))
                
        return max(versions) if versions else 0

    def unlock_scales(self):
        v = self.current_model_version
        reply = QMessageBox.question(self, 'Unlock Model Parameters',
                                     f"Unlocking scales will delete the current trained model (HydroBound-ML_v{v}.joblib). Your manual samples will NOT be deleted. Proceed?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            model_path = os.path.join(self.model_dir, f"HydroBound-ML_v{v}.joblib")
            if os.path.exists(model_path):
                os.remove(model_path)
            self.spin_scale_1.setEnabled(True)
            self.spin_scale_2.setEnabled(True)
            self.spin_scale_3.setEnabled(True)
            self.btn_unlock.setEnabled(False)
            self.version_label.setText(f"Model Version: None (Found Samples v{v})")
            QMessageBox.information(self, "Unlocked", "Parameters unlocked. Model deleted. Please retrain.")

    def update_sample_counter(self):
        water_count = sum(1 for c in self.session_clicks if c[2] == 1)
        non_water_count = sum(1 for c in self.session_clicks if c[2] == 0)
        self.sample_label.setText(f"Pending ML Clicks -> Water: {water_count} | Non-Water: {non_water_count}")

    def clear_manual_samples_ui(self):
        for mx, my, label, l_obj in self.session_clicks:
            try:
                l_obj.remove()
            except Exception:
                pass
        self.session_clicks.clear()
        self.samples_X.clear()
        self.samples_y.clear()
        self.sample_coords.clear()
        self.update_sample_counter()
        self.canvas.draw_idle()

    def on_limits_changed(self, ax=None):
        if not self._pan_active:
            self.render_timer.start(150)

    def find_layer_path(self, layer_name):
        candidates = ["DTM_ellipsoidal_heights.tif", "dtm_ellipsoidal_heights.tif", "DTM.tif", "dtm.tif",
                      "DTM.TIF"] if layer_name == "dtm" else [f"{layer_name.upper()}.tif", f"{layer_name}.tif",
                                                              f"{layer_name}.TIF"]
        for c in candidates:
            full_p = os.path.join(self.input_dir, c)
            if os.path.exists(full_p):
                return full_p
        return None

    def get_layer_src(self, layer_name):
        if layer_name in self.data_srcs:
            return self.data_srcs[layer_name]
        p = self.find_layer_path(layer_name)
        if p:
            src = rasterio.open(p)
            self.data_srcs[layer_name] = src

            if layer_name not in self.layer_norms and layer_name not in ['ortho_rgb', 'ortho_nir']:
                ov_factors = src.overviews(1)
                if ov_factors:
                    dec = ov_factors[-1]
                    sample = src.read(1, out_shape=(int(src.height // dec), int(src.width // dec)))
                else:
                    cx, cy = src.width // 2, src.height // 2
                    w = 2000
                    win = rasterio.windows.Window(max(0, cx - w // 2), max(0, cy - w // 2), min(src.width, w),
                                                  min(src.height, w))
                    sample = src.read(1, window=win)

                if src.dtypes[0] == 'int16' and layer_name in ['ndwi', 'ndvi', 'ngrdi']:
                    sample = sample.astype(float) / 10000.0

                valid = sample[np.isfinite(sample) & (sample > -100)] if layer_name == 'dtm' else sample[
                    np.isfinite(sample) & (sample >= -1.0) & (sample <= 1.0)]
                if len(valid) > 0:
                    vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
                    if vmin >= vmax:
                        vmin -= 0.01
                        vmax += 0.01
                    self.layer_norms[layer_name] = Normalize(vmin=vmin, vmax=vmax)
                elif layer_name in ['ndwi', 'ndvi', 'ngrdi']:
                    self.layer_norms[layer_name] = Normalize(vmin=-1.0, vmax=1.0)
            self.update_layer_ui()
            return src
        return None

    def update_layer_ui(self):
        for name, rb in self.layer_buttons.items():
            if name in self.data_srcs:
                rb.setText(f"[RAM] {name.upper()}")
                rb.setEnabled(True)
                rb.setStyleSheet("color: darkgreen; font-weight: normal;")
            elif self.find_layer_path(name):
                rb.setText(f"[DISK] {name.upper()}")
                rb.setEnabled(True)
                rb.setStyleSheet("color: black; font-weight: normal;")
            else:
                rb.setText(f"[MISSING] {name.upper()}")
                rb.setEnabled(False)
                rb.setStyleSheet("color: gray;")

        dtm_exists = self.find_layer_path("dtm") is not None
        if not dtm_exists:
            self.check_export_grid.setChecked(False)
            self.check_export_grid.setEnabled(False)
            self.check_export_grid.setText("Grid Export Unavailable (Missing DTM)")
            self.combo_z_method.setEnabled(False) 
        else:
            self.check_export_grid.setEnabled(True)
            self.check_export_grid.setText("Enable Grid Export (Requires DTM)")
            self.combo_z_method.setEnabled(True) 

    def check_and_generate_indices(self):
        required = ["ndwi", "ndvi", "ngrdi"]
        missing = [idx for idx in required if not self.find_layer_path(idx)]

        if not missing:
            return True

        needs_nir = "ndvi" in missing or "ndwi" in missing

        rgb_path = self.find_layer_path("ortho_rgb")
        nir_path = self.find_layer_path("ortho_nir") if needs_nir else None

        missing_core_msg = []
        if not rgb_path: missing_core_msg.append("'ortho_rgb'")
        if needs_nir and not nir_path: missing_core_msg.append("'ortho_nir'")

        if missing_core_msg:
            QMessageBox.warning(
                self,
                "Missing Core Data",
                f"The following required indices are missing: {', '.join(missing).upper()}.\n\n"
                f"However, the required source data ({' and '.join(missing_core_msg)}) is also missing from the INPUT folder.\n"
                "Indices cannot be generated automatically. Please provide the required data."
            )
            return False

        reply = QMessageBox.question(
            self,
            'Missing Indices Detected',
            f"The following required indices are missing from the INPUT directory:\n{', '.join(missing).upper()}\n\n"
            "Would you like the application to generate them now from the available orthomosaics? "
            "(Selecting 'No' will abort the loading process).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply == QMessageBox.StandardButton.No:
            return False

        try:
            with rasterio.open(rgb_path) as tmp_rgb:
                rgb_bands = tmp_rgb.count
            nir_bands = None
            if needs_nir:
                with rasterio.open(nir_path) as tmp_nir:
                    nir_bands = tmp_nir.count
        except Exception as e:
            QMessageBox.critical(self, "Read Error", f"Failed to read source rasters: {e}")
            return False

        band_dialog = BandSelectionDialog(rgb_bands, nir_bands, self)
        if band_dialog.exec() == QDialog.DialogCode.Rejected:
            return False

        mapping = band_dialog.get_mapping()

        p = QProgressDialog(f"Generating Missing Indices: {', '.join(missing).upper()}...", "Cancel", 0, 100, self)
        p.setWindowTitle("Processing Data")
        p.setWindowModality(Qt.WindowModality.ApplicationModal)
        p.show()
        QCoreApplication.processEvents()

        src_rgb, src_nir, nir_vrt = None, None, None
        try:
            with rasterio.Env():
                src_rgb = rasterio.open(rgb_path)

                if needs_nir:
                    src_nir = rasterio.open(nir_path)
                    vrt_options = {
                        'crs': src_rgb.crs,
                        'transform': src_rgb.transform,
                        'height': src_rgb.height,
                        'width': src_rgb.width,
                        'resampling': Resampling.nearest
                    }
                    nir_vrt = WarpedVRT(src_nir, **vrt_options)

                meta = src_rgb.meta.copy()
                meta.update(dtype=rasterio.float32, count=1, compress='lzw', tiled=True, nodata=-1.5)
                meta.update(blockxsize=512, blockysize=512)

                dst_files = {idx: rasterio.open(os.path.join(self.input_dir, f"{idx.upper()}.tif"), 'w', **meta) for idx
                             in missing}

                chunk_size = 4096
                windows = []
                for row_off in range(0, src_rgb.height, chunk_size):
                    for col_off in range(0, src_rgb.width, chunk_size):
                        w = min(chunk_size, src_rgb.width - col_off)
                        h = min(chunk_size, src_rgb.height - row_off)
                        windows.append(rasterio.windows.Window(col_off, row_off, w, h))

                rgb_nodata = src_rgb.nodata
                nir_nodata = src_nir.nodata if needs_nir else None
                total_windows = len(windows)

                for i, window in enumerate(windows):
                    if p.wasCanceled():
                        for dst in dst_files.values():
                            dst.close()
                        return False

                    red = src_rgb.read(mapping['red'], window=window).astype('float32')
                    green = src_rgb.read(mapping['green'], window=window).astype('float32')

                    valid_mask = np.ones(red.shape, dtype=bool)
                    if rgb_nodata is not None:
                        valid_mask &= (red != rgb_nodata) & (green != rgb_nodata)

                    if needs_nir:
                        nir = nir_vrt.read(mapping['nir'], window=window).astype('float32')
                        if nir_nodata is not None:
                            valid_mask &= (nir != nir_nodata)

                    np.seterr(divide='ignore', invalid='ignore')

                    if "ngrdi" in missing:
                        denom = green + red
                        calc_mask = valid_mask & (denom != 0)
                        out_arr = np.full(red.shape, -1.5, dtype='float32')
                        out_arr[calc_mask] = (green[calc_mask] - red[calc_mask]) / denom[calc_mask]
                        dst_files["ngrdi"].write(out_arr, 1, window=window)

                    if "ndwi" in missing:
                        denom = green + nir
                        calc_mask = valid_mask & (denom != 0)
                        out_arr = np.full(red.shape, -1.5, dtype='float32')
                        out_arr[calc_mask] = (green[calc_mask] - nir[calc_mask]) / denom[calc_mask]
                        dst_files["ndwi"].write(out_arr, 1, window=window)

                    if "ndvi" in missing:
                        denom = nir + red
                        calc_mask = valid_mask & (denom != 0)
                        out_arr = np.full(red.shape, -1.5, dtype='float32')
                        out_arr[calc_mask] = (nir[calc_mask] - red[calc_mask]) / denom[calc_mask]
                        dst_files["ndvi"].write(out_arr, 1, window=window)

                    if total_windows > 0 and i % max(1, (total_windows // 100)) == 0:
                        p.setValue(int((i / total_windows) * 100))
                        QCoreApplication.processEvents()

                if nir_vrt: nir_vrt.close()
                if src_nir: src_nir.close()
                src_rgb.close()

                p.setLabelText("Building Overviews (Pyramids)... This may take a minute, please wait.")
                QCoreApplication.processEvents()

                factors = [2, 4, 8, 16, 32, 64]

                for dst in dst_files.values():
                    dst.build_overviews(factors, Resampling.nearest)
                    dst.update_tags(ns='rio_overview', resampling='nearest')
                    dst.close()

            del dst_files
            gc.collect()
            time.sleep(1.0)

            return True
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", f"Failed to generate indices: {e}")
            return False
        finally:
            p.close()

    def load_project(self):
        path = QFileDialog.getExistingDirectory(self, "Select Main Project Directory")
        if not path:
            return

        for src in self.data_srcs.values():
            try:
                src.close()
            except:
                pass
        self.data_srcs.clear()
        self.layer_norms.clear()

        self.clear_manual_samples_ui()
        self.clear_measurements()
        self.clear_edit_drawing()

        if self.img_artist:
            try:
                self.img_artist.remove()
            except:
                pass
            self.img_artist = None

        if self.vector_artist:
            try:
                self.vector_artist.remove()
            except:
                pass
            self.vector_artist = None

        if self.aoi_artist:
            try:
                self.aoi_artist.remove()
            except:
                pass
            self.aoi_artist = None

        self.vector_mask = None
        self.vector_lines_cache = []
        self.aoi_lines_cache = []
        self.current_model_version = 0
        self.version_label.setText("Model Version: None")

        self.ax.clear()
        self.canvas.draw_idle()

        self.base_dir = path
        self.input_dir = os.path.join(path, "INPUT")
        if not os.path.exists(self.input_dir):
            QMessageBox.critical(self, "Directory Error", "Selected directory does not contain an 'INPUT' folder.")
            return

        self.output_dir = os.path.join(path, "OUTPUT")
        self.model_dir = os.path.join(path, "MODEL")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        if not self.check_and_generate_indices():
            QMessageBox.information(self, "Project Load Aborted", "Project loading was aborted due to missing indices.")
            return

        kml_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.kml') and 'aoi' in f.lower()]
        if not kml_files:
            kml_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.kml')]

        self.selected_kml_path = ""
        if len(kml_files) > 1:
            item, ok = QInputDialog.getItem(self, "Select AOI File",
                                            "Multiple KML files found.\nPlease select the primary AOI file:", kml_files,
                                            0, False)
            if ok and item:
                self.selected_kml_path = os.path.join(self.input_dir, item)
            else:
                QMessageBox.warning(self, "Warning", "AOI selection cancelled. Project will load without KML boundary.")
        elif len(kml_files) == 1:
            self.selected_kml_path = os.path.join(self.input_dir, kml_files[0])

        metadata_path = os.path.join(self.input_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.project_metadata = json.load(f)
            except Exception:
                self.project_metadata = None
        else:
            self.project_metadata = None

        self.update_layer_ui()

        self.current_model_version = self.find_latest_model_version()
        model_path = os.path.join(self.model_dir, f"HydroBound-ML_v{self.current_model_version}.joblib")

        if self.current_model_version > 0 and os.path.exists(model_path):
            self.version_label.setText(f"Model Version: v{self.current_model_version}")
            try:
                saved = joblib.load(model_path)
                if 'scales' in saved:
                    sc = saved['scales']
                    self.spin_scale_1.setValue(sc[0])
                    self.spin_scale_2.setValue(sc[1])
                    self.spin_scale_3.setValue(sc[2])
                    self.spin_scale_1.setEnabled(False)
                    self.spin_scale_2.setEnabled(False)
                    self.spin_scale_3.setEnabled(False)
                    self.btn_unlock.setEnabled(True)
            except Exception:
                pass
        elif self.current_model_version > 0 and not os.path.exists(model_path):
            self.version_label.setText(f"Model Version: None (Found Samples v{self.current_model_version})")
            self.spin_scale_1.setEnabled(True)
            self.spin_scale_2.setEnabled(True)
            self.spin_scale_3.setEnabled(True)
            self.btn_unlock.setEnabled(False)
        else:
            self.version_label.setText("Model Version: None (New Project)")
            self.spin_scale_1.setEnabled(True)
            self.spin_scale_2.setEnabled(True)
            self.spin_scale_3.setEnabled(True)
            self.btn_unlock.setEnabled(False)

        otsu_cache_path = os.path.join(self.model_dir, "HydroBound-ML_otsu_samples_v1.0.0.joblib")
        if os.path.exists(otsu_cache_path):
            try:
                data = joblib.load(otsu_cache_path)
                if 'version' not in data or data['version'] != '1.0.0':
                    os.remove(otsu_cache_path)
            except Exception:
                pass

        p = QProgressDialog("Initializing Project...", "Cancel", 0, 100, self)
        p.setWindowModality(Qt.WindowModality.ApplicationModal)
        p.show()
        QCoreApplication.processEvents()

        try:
            ndwi_src = self.get_layer_src("ndwi")
            if ndwi_src:
                self.layer_buttons["ndwi"].setChecked(True)
                self.current_layer = "ndwi"
            elif self.get_layer_src("ortho_rgb"):
                self.layer_buttons["ortho_rgb"].setChecked(True)
                self.current_layer = "ortho_rgb"

            if self.find_layer_path("dtm"):
                self.get_layer_src("dtm")

            src_to_extent = self.get_layer_src(self.current_layer)

            if src_to_extent:
                self.full_extent = [src_to_extent.bounds.left, src_to_extent.bounds.right, src_to_extent.bounds.bottom,
                                    src_to_extent.bounds.top]
                self._is_updating = True
                self.ax.clear()

                self.img_artist = self.ax.imshow(np.zeros((10, 10)), origin='upper', extent=self.full_extent, zorder=1)
                self.vector_artist = LineCollection([], color='magenta', linewidth=2.0, zorder=5)
                self.ax.add_collection(self.vector_artist)

                self.aoi_artist = LineCollection([], color='gold', linewidth=1.5, linestyle='--', zorder=10)
                self.ax.add_collection(self.aoi_artist)

                self.ax.set_xlim(self.full_extent[0], self.full_extent[1])
                self.ax.set_ylim(self.full_extent[2], self.full_extent[3])

                if self.selected_kml_path and os.path.exists(self.selected_kml_path):
                    try:
                        aoi_gdf = gpd.read_file(self.selected_kml_path, driver='KML')
                        if aoi_gdf.crs is None:
                            aoi_gdf.set_crs(epsg=4326, inplace=True)
                        if aoi_gdf.crs != src_to_extent.crs:
                            aoi_gdf = aoi_gdf.to_crs(src_to_extent.crs)

                        self.aoi_lines_cache = []
                        for geom in aoi_gdf.geometry:
                            if geom is None:
                                continue
                            if geom.geom_type == 'Polygon':
                                self.aoi_lines_cache.append(np.column_stack(geom.exterior.xy))
                                for interior in geom.interiors:
                                    self.aoi_lines_cache.append(np.column_stack(interior.xy))
                            elif geom.geom_type == 'MultiPolygon':
                                for poly in geom.geoms:
                                    self.aoi_lines_cache.append(np.column_stack(poly.exterior.xy))
                                    for interior in poly.interiors:
                                        self.aoi_lines_cache.append(np.column_stack(interior.xy))

                        self.aoi_artist.set_segments(self.aoi_lines_cache)
                        self.check_aoi.setChecked(True)
                    except Exception as e:
                        print(f"Failed to load AOI: {e}")

                self._is_updating = False
                self.toolbar.update()
                self.toolbar.push_current()
                self.fetch_high_res_raster()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load failed: {e}")
        finally:
            p.close()

    def change_layer(self, id):
        names = list(self.layer_buttons.keys())
        if not self.get_layer_src(names[id]):
            self.layer_buttons[self.current_layer].setChecked(True)
            return
        self.current_layer = names[id]
        self._is_updating = False
        self.fetch_high_res_raster()

    def toggle_vector_visibility(self):
        if self.vector_artist:
            self.vector_artist.set_visible(self.check_mask.isChecked())
            self.canvas.draw_idle()

    def toggle_aoi_visibility(self):
        if self.aoi_artist:
            self.aoi_artist.set_visible(self.check_aoi.isChecked())
            self.canvas.draw_idle()

    def _cache_vector_lines(self, gdf):
        self.vector_lines_cache = []
        if gdf is not None and not gdf.empty:
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type == 'Polygon':
                    self.vector_lines_cache.append(np.column_stack(geom.exterior.xy))
                    for interior in geom.interiors:
                        self.vector_lines_cache.append(np.column_stack(interior.xy))
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        self.vector_lines_cache.append(np.column_stack(poly.exterior.xy))
                        for interior in poly.interiors:
                            self.vector_lines_cache.append(np.column_stack(interior.xy))
        if self.vector_artist:
            self.vector_artist.set_segments(self.vector_lines_cache)
            self.canvas.draw_idle()

    def fetch_high_res_raster(self):
        if self._pan_active:
            return
        src = self.get_layer_src(self.current_layer)
        if not src or self.img_artist is None:
            return

        self._is_updating = True
        cur_xl, cur_yl = self.ax.get_xlim(), self.ax.get_ylim()
        dx, dy = cur_xl[1] - cur_xl[0], cur_yl[1] - cur_yl[0]
        margin_x, margin_y = abs(dx) * 0.2, abs(dy) * 0.2
        rx_min, rx_max = min(cur_xl) - margin_x, max(cur_xl) + margin_x
        ry_min, ry_max = min(cur_yl) - margin_y, max(cur_yl) + margin_y

        try:
            win = from_bounds(rx_min, ry_min, rx_max, ry_max, src.transform)
            count = src.count
            data = src.read(list(range(1, count + 1)), window=win, out_shape=(count, 800, 800),
                            resampling=rasterio.enums.Resampling.nearest, boundless=True, fill_value=src.nodata or 0)

            if src.dtypes[0] == 'int16' and self.current_layer in ['ndwi', 'ndvi', 'ngrdi']:
                data = data.astype('float32') / 10000.0
                if src.nodata is not None:
                    data = np.where(np.isclose(data, src.nodata / 10000.0), np.nan, data)
            elif src.nodata is not None and count < 3:
                data = np.where(np.isclose(data, src.nodata), np.nan, data)

            if count >= 3:
                data_disp = np.transpose(data[:3], (1, 2, 0)).astype(np.uint8)
                self.img_artist.set_data(data_disp)
                self.img_artist.set_cmap(None)
                self.img_artist.set_norm(None)
            else:
                norm = self.layer_norms.get(self.current_layer)
                self.img_artist.set_data(data[0])
                self.img_artist.set_cmap(self.cmaps.get(self.current_layer))
                if norm:
                    self.img_artist.set_norm(norm)
                    self.img_artist.set_clim(vmin=norm.vmin, vmax=norm.vmax)
                else:
                    self.img_artist.set_norm(None)
                    if self.current_layer in ['ndwi', 'ndvi', 'ngrdi']:
                        self.img_artist.set_clim(vmin=-1.0, vmax=1.0)

            self.img_artist.set_extent([rx_min, rx_max, ry_min, ry_max])
            self.ax.set_aspect('equal')
            self.canvas.draw_idle()
        except Exception:
            pass
        finally:
            self._is_updating = False

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        scale = 0.8 if event.button == 'up' else 1.25
        cur_x, cur_y = self.ax.get_xlim(), self.ax.get_ylim()
        x_range, y_range = cur_x[1] - cur_x[0], cur_y[1] - cur_y[0]
        rel_x, rel_y = (event.xdata - cur_x[0]) / x_range, (event.ydata - cur_y[0]) / y_range
        new_width, new_height = x_range * scale, y_range * scale

        self.ax.set_xlim([event.xdata - new_width * rel_x, event.xdata + new_width * (1 - rel_x)])
        self.ax.set_ylim([event.ydata - new_height * rel_y, event.ydata + new_height * (1 - rel_y)])
        self.canvas.draw_idle()
        self.render_timer.start(150)

    def on_mouse_move(self, event):
        if self._pan_active and event.inaxes == self.ax:
            x0, y0 = self.ax.transData.inverted().transform((0, 0))
            x1, y1 = self.ax.transData.inverted().transform((event.x - self._pan_start_x, event.y - self._pan_start_y))
            dx, dy = x1 - x0, y1 - y0
            self.ax.set_xlim(self._pan_start_xlim[0] - dx, self._pan_start_xlim[1] - dx)
            self.ax.set_ylim(self._pan_start_ylim[0] - dy, self._pan_start_ylim[1] - dy)
            self.canvas.draw_idle()

    def on_mouse_release(self, event):
        if event.button == 2 or self.toolbar.mode != '':
            self._pan_active = False
            self.render_timer.start(150)

    def get_multiscale_features(self, mx, my, src_ndwi):
        feats = []
        res_x = abs(src_ndwi.transform[0])
        scales = [self.spin_scale_1.value(), self.spin_scale_2.value(), self.spin_scale_3.value()]

        src_ndvi = self.get_layer_src("ndvi")
        src_ngrdi = self.get_layer_src("ngrdi")

        if not src_ndvi or not src_ngrdi:
            return None

        for src in [src_ndwi, src_ndvi, src_ngrdi]:
            val = list(src.sample([(mx, my)]))[0][0]
            if src.dtypes[0] == 'int16':
                val = val / 10000.0
            feats.append(float(val))

        row, col = src_ndwi.index(mx, my)

        def compute_std(arr):
            valid = arr[np.isfinite(arr)]
            if valid.size == 0:
                return 0.0
            mean_sq = np.mean(valid ** 2)
            sq_mean = np.mean(valid) ** 2
            return float(np.sqrt(np.clip(mean_sq - sq_mean, 0, None)))

        for idx, m_scale in enumerate(scales):
            s_px = max(1, int(round(m_scale / res_x)))
            if s_px % 2 == 0:
                s_px += 1
            h_px = s_px // 2

            win = rasterio.windows.Window(col - h_px, row - h_px, s_px, s_px)

            arr_ndwi = src_ndwi.read(1, window=win, boundless=True, fill_value=np.nan)
            if src_ndwi.dtypes[0] == 'int16':
                arr_ndwi = arr_ndwi.astype(float) / 10000.0
            feats.append(compute_std(arr_ndwi))

            arr_ndvi = src_ndvi.read(1, window=win, boundless=True, fill_value=np.nan)
            if src_ndvi.dtypes[0] == 'int16':
                arr_ndvi = arr_ndvi.astype(float) / 10000.0
            feats.append(compute_std(arr_ndvi))

            arr_ngrdi = src_ngrdi.read(1, window=win, boundless=True, fill_value=np.nan)
            if src_ngrdi.dtypes[0] == 'int16':
                arr_ngrdi = arr_ngrdi.astype(float) / 10000.0
            feats.append(compute_std(arr_ngrdi))

            if idx < 2:
                sig = max(1.0, s_px / 3.0)
                pad = int(math.ceil(sig * 3))
                win_grad = rasterio.windows.Window(col - pad, row - pad, pad * 2 + 1, pad * 2 + 1)
                arr_grad = src_ndwi.read(1, window=win_grad, boundless=True, fill_value=np.nan)
                if src_ndwi.dtypes[0] == 'int16':
                    arr_grad = arr_grad.astype(float) / 10000.0
                grad_val = ndimage.gaussian_gradient_magnitude(np.nan_to_num(arr_grad, 0), sigma=sig)[pad, pad]
                feats.append(float(grad_val))

        return feats

    def on_click(self, event):
        if event.button == 2:
            if event.inaxes == self.ax:
                self._pan_active = True
                self._pan_start_x, self._pan_start_y = event.x, event.y
                self._pan_start_xlim, self._pan_start_ylim = self.ax.get_xlim(), self.ax.get_ylim()
            return
        if event.inaxes != self.ax or self.toolbar.mode != '':
            return

        if hasattr(self, 'btn_measure') and self.btn_measure.isChecked() and event.button == 1:
            self._measure_points.append((event.xdata, event.ydata))
            pt, = self.ax.plot(event.xdata, event.ydata, 'yo', ms=6, markeredgecolor='black', zorder=30)
            self._measure_artists.append(pt)

            if len(self._measure_points) == 2:
                x1, y1 = self._measure_points[0]
                x2, y2 = self._measure_points[1]
                dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                ln, = self.ax.plot([x1, x2], [y1, y2], 'y--', lw=2, zorder=30)
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                txt = self.ax.text(mid_x, mid_y, f"{dist:.2f} m", color='black',
                                   bbox=dict(facecolor='yellow', alpha=0.7), zorder=31, ha='center', va='center')

                self._measure_artists.extend([ln, txt])
                self._measure_points = []
            self.canvas.draw_idle()
            return

        if self.edit_mode is not None:
            if self.vector_mask is None or self.vector_mask.empty:
                QMessageBox.warning(self, "Warning", "No vector mask to edit. Please run Predict first.")
                return

            click_point = Point(event.xdata, event.ydata)

            if self.edit_mode == 'del_poly' and event.button == 1:
                mask = self.vector_mask.geometry.contains(click_point)
                if mask.any():
                    self.vector_mask = self.vector_mask[~mask]
                    self._cache_vector_lines(self.vector_mask)
                return

            elif self.edit_mode == 'fill_hole' and event.button == 1:
                new_geoms = []
                hole_filled = False
                for geom in self.vector_mask.geometry:
                    if geom.geom_type == 'Polygon':
                        if box(*geom.bounds).contains(click_point):
                            new_interiors = []
                            for interior in geom.interiors:
                                hole_poly = Polygon(interior)
                                if hole_poly.contains(click_point):
                                    hole_filled = True
                                else:
                                    new_interiors.append(interior)
                            new_geoms.append(Polygon(geom.exterior, new_interiors))
                        else:
                            new_geoms.append(geom)

                    elif geom.geom_type == 'MultiPolygon':
                        new_parts = []
                        for poly in geom.geoms:
                            if box(*poly.bounds).contains(click_point):
                                new_interiors = []
                                for interior in poly.interiors:
                                    hole_poly = Polygon(interior)
                                    if hole_poly.contains(click_point):
                                        hole_filled = True
                                    else:
                                        new_interiors.append(interior)
                                new_parts.append(Polygon(poly.exterior, new_interiors))
                            else:
                                new_parts.append(poly)
                        new_geoms.append(MultiPolygon(new_parts))

                if hole_filled:
                    self.vector_mask = gpd.GeoDataFrame({'geometry': new_geoms}, crs=self.vector_mask.crs)
                    self._cache_vector_lines(self.vector_mask)
                return

            elif self.edit_mode in ['add_poly', 'cut_poly']:
                if event.button == 1:
                    self.edit_drawing_points.append((event.xdata, event.ydata))
                    pt, = self.ax.plot(event.xdata, event.ydata, 'ro', ms=4, zorder=35)
                    self.edit_drawing_artists.append(pt)

                    if len(self.edit_drawing_points) > 1:
                        x_vals = [p[0] for p in self.edit_drawing_points]
                        y_vals = [p[1] for p in self.edit_drawing_points]
                        ln, = self.ax.plot(x_vals, y_vals, 'r-', lw=1.5, zorder=35)
                        self.edit_drawing_artists.append(ln)

                    self.canvas.draw_idle()

                elif event.button == 3:
                    if len(self.edit_drawing_points) >= 3:
                        drawn_poly = Polygon(self.edit_drawing_points)
                        if not drawn_poly.is_valid:
                            drawn_poly = drawn_poly.buffer(0)

                        drawn_gdf = gpd.GeoDataFrame({'geometry': [drawn_poly]}, crs=self.vector_mask.crs)

                        try:
                            if self.edit_mode == 'add_poly':
                                self.vector_mask = gpd.overlay(self.vector_mask, drawn_gdf, how='union')
                                self.vector_mask = gpd.GeoDataFrame({'geometry': [self.vector_mask.unary_union]},
                                                                    crs=self.vector_mask.crs)
                                self.vector_mask = self.vector_mask.explode(index_parts=False).reset_index(drop=True)

                            elif self.edit_mode == 'cut_poly':
                                self.vector_mask = gpd.overlay(self.vector_mask, drawn_gdf, how='difference')
                                self.vector_mask = self.vector_mask.explode(index_parts=False).reset_index(drop=True)

                            self._cache_vector_lines(self.vector_mask)
                        except Exception as e:
                            QMessageBox.warning(self, "Topology Error", f"Could not perform operation: {e}")

                    self.clear_edit_drawing()
                return

        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.KeyboardModifier.ControlModifier and event.button == 1:
            if not self.session_clicks:
                return
            min_dist, min_idx = float('inf'), -1
            for i, (px, py, label, l_obj) in enumerate(self.session_clicks):
                dist = (px - event.xdata) ** 2 + (py - event.ydata) ** 2
                if dist < min_dist:
                    min_dist, min_idx = dist, i
            x_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
            if np.sqrt(min_dist) < (x_range * 0.05):
                _, _, _, l_obj = self.session_clicks.pop(min_idx)
                l_obj.remove()
                self.update_sample_counter()
                self.canvas.draw_idle()
            return

        if event.button not in [1, 3]:
            return
        label = 1 if event.button == 1 else 0
        color = 'lime' if label == 1 else 'red'

        try:
            line = \
            self.ax.plot(event.xdata, event.ydata, 'o', color=color, ms=8, markeredgecolor='white', markeredgewidth=1.2,
                         zorder=20)[0]
            self.session_clicks.append((event.xdata, event.ydata, label, line))
            self.update_sample_counter()
            self.canvas.draw_idle()
        except Exception as e:
            print(f"Sampling GUI error: {e}")

    def train_model(self):
        v = self.current_model_version
        manual_path = os.path.join(self.model_dir, f"HydroBound-ML_v{v}_manual_samples.joblib") if v > 0 else ""

        if not self.session_clicks and not (manual_path and os.path.exists(manual_path)):
            QMessageBox.warning(self, "Missing Data", "Add manual points on the map first.")
            return

        p = QProgressDialog("Extracting spatial features for manual clicks...", "Cancel", 0, 100, self)
        p.setWindowModality(Qt.WindowModality.ApplicationModal)
        p.setFixedSize(500, 150)
        p.show()
        QCoreApplication.processEvents()

        try:
            self.samples_X, self.samples_y, self.sample_coords = [], [], []
            if self.session_clicks:
                src_ndwi = self.get_layer_src("ndwi")
                if not src_ndwi:
                    raise Exception("NDWI raster required for feature extraction.")

                jitter_m = self.spin_jitter.value()
                jitter_pts = self.spin_jitter_points.value()
                total_clicks = len(self.session_clicks)

                for idx, (mx, my, label, _) in enumerate(self.session_clicks):
                    if p.wasCanceled():
                        return

                    offsets = [(0, 0)]
                    if jitter_m > 0 and jitter_pts > 0:
                        for i in range(jitter_pts):
                            angle = 2 * np.pi * i / jitter_pts
                            dx = np.cos(angle) * jitter_m
                            dy = np.sin(angle) * jitter_m
                            offsets.append((dx, dy))

                    for dx, dy in offsets:
                        sx, sy = mx + dx, my + dy
                        feats = self.get_multiscale_features(sx, sy, src_ndwi)
                        if feats is not None:
                            self.samples_X.append(feats)
                            self.samples_y.append(label)
                            self.sample_coords.append((sx, sy))

                    p.setValue(int(30 * (idx / total_clicks)))
                    QCoreApplication.processEvents()

            p.setLabelText("Merging with previous version...")
            QCoreApplication.processEvents()

            has_new_samples = len(self.samples_X) > 0
            if has_new_samples:
                X_manual_current = np.array(self.samples_X)
                y_manual_current = np.array(self.samples_y)
                coords_manual_current = np.array(self.sample_coords)

            if v > 0 and os.path.exists(manual_path):
                saved_manual = joblib.load(manual_path)
                if has_new_samples:
                    X_manual_all = np.vstack((saved_manual['X'], X_manual_current))
                    y_manual_all = np.concatenate((saved_manual['y'], y_manual_current))
                    coords_manual_all = np.vstack((saved_manual['coords'], coords_manual_current))
                else:
                    X_manual_all = saved_manual['X']
                    y_manual_all = saved_manual['y']
                    coords_manual_all = saved_manual['coords']
            else:
                X_manual_all, y_manual_all, coords_manual_all = X_manual_current, y_manual_current, coords_manual_current

            new_v = v + 1
            new_manual_path = os.path.join(self.model_dir, f"HydroBound-ML_v{new_v}_manual_samples.joblib")
            joblib.dump(
                {'X': X_manual_all, 'y': y_manual_all, 'coords': coords_manual_all, 'version': f'1.0.7_v{new_v}'},
                new_manual_path)

            baseline_path = os.path.join(self.model_dir, "HydroBound-ML_otsu_samples_v1.0.0.joblib")
            filtered_base_len = 0

            if os.path.exists(baseline_path):
                base = joblib.load(baseline_path)
                X_base, y_base, coords_base = base['X'], base['y'], base.get('coords', None)

                ndwi_base = X_base[:, 0]
                w_min = self.spin_otsu_water_min.value()
                l_max = self.spin_otsu_land_max.value()

                core_water = (y_base == 1) & (ndwi_base >= w_min)
                core_land = (y_base == 0) & (ndwi_base <= l_max)
                core_mask = core_water | core_land

                X_base = X_base[core_mask]
                y_base = y_base[core_mask]
                if coords_base is not None:
                    coords_base = coords_base[core_mask]

                filtered_base_len = len(X_base)

                otsu_limit = self.spin_otsu_base.value()
                if len(X_base) > otsu_limit:
                    idx = np.random.choice(len(X_base), otsu_limit, replace=False)
                    X_base = X_base[idx]
                    y_base = y_base[idx]
                    coords_base = coords_base[idx] if coords_base is not None else None

                p.setLabelText("Purging conflicting Otsu errors...")
                p.setValue(50)
                QCoreApplication.processEvents()

                deleted_points = 0
                if coords_base is not None and len(coords_base) > 0:
                    tree = cKDTree(coords_base)
                    to_delete = set()
                    for i, (mx, my) in enumerate(coords_manual_all):
                        idx_near = tree.query_ball_point([mx, my], r=100.0)
                        for idx in idx_near:
                            if y_base[idx] != y_manual_all[i]:
                                to_delete.add(idx)
                    if to_delete:
                        mask = np.ones(len(y_base), dtype=bool)
                        mask[list(to_delete)] = False
                        X_base = X_base[mask]
                        y_base = y_base[mask]
                        coords_base = coords_base[mask]
                        deleted_points = len(to_delete)

                if len(X_base) > 5:
                    X_b_tr, X_b_val, y_b_tr, y_b_val = train_test_split(X_base, y_base, test_size=0.2, random_state=42)
                else:
                    X_b_tr, X_b_val, y_b_tr, y_b_val = X_base, X_base, y_base, y_base
            else:
                deleted_points = 0
                filtered_base_len = 0
                w_min, l_max = self.spin_otsu_water_min.value(), self.spin_otsu_land_max.value()
                X_b_tr, X_b_val, y_b_tr, y_b_val = [], [], [], []

            if len(X_manual_all) > 5:
                p.setLabelText("Grouping manual samples to prevent data leakage...")
                QCoreApplication.processEvents()
                
                eps_distance = max(1.0, self.spin_jitter.value() * 2.5)
                clustering = DBSCAN(eps=eps_distance, min_samples=1).fit(coords_manual_all)
                groups_manual = clustering.labels_
                
                num_groups = len(np.unique(groups_manual))
                
                if num_groups > 5:
                    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                    train_idx, val_idx = next(gss.split(X_manual_all, y_manual_all, groups_manual))
                    
                    X_m_tr, X_m_val = X_manual_all[train_idx], X_manual_all[val_idx]
                    y_m_tr, y_m_val = y_manual_all[train_idx], y_manual_all[val_idx]
                else:
                    X_m_tr, X_m_val, y_m_tr, y_m_val = train_test_split(X_manual_all, y_manual_all, test_size=0.2, random_state=42)
            else:
                X_m_tr, X_m_val, y_m_tr, y_m_val = X_manual_all, X_manual_all, y_manual_all, y_manual_all

            repeats = max(1, self.spin_clones.value())
            noise_std = self.spin_noise_aug.value()

            if repeats > 1:
                X_aug_list = [X_m_tr]
                y_aug_list = [y_m_tr]
                for _ in range(repeats - 1):
                    noise = np.random.normal(0, noise_std, X_m_tr.shape)
                    X_aug_list.append(X_m_tr + noise)
                    y_aug_list.append(y_m_tr)
                X_m_tr_expanded = np.vstack(X_aug_list)
                y_m_tr_expanded = np.concatenate(y_aug_list)
            else:
                X_m_tr_expanded = X_m_tr
                y_m_tr_expanded = y_m_tr

            if len(X_b_tr) > 0:
                X_train = np.vstack((X_b_tr, X_m_tr_expanded))
                y_train = np.concatenate((y_b_tr, y_m_tr_expanded))
                w_train = np.concatenate(
                    (np.ones(len(y_b_tr)), np.full(len(y_m_tr_expanded), self.spin_weight.value())))
            else:
                X_train = X_m_tr_expanded
                y_train = y_m_tr_expanded
                w_train = np.full(len(y_m_tr_expanded), self.spin_weight.value())

            if len(X_b_val) > 0:
                X_eval = np.vstack((X_b_val, X_m_val))
                y_eval = np.concatenate((y_b_val, y_m_val))
            else:
                X_eval = X_m_val
                y_eval = y_m_val

            p.setLabelText(f"Training Model Version v{new_v}...")
            p.setValue(70)
            QCoreApplication.processEvents()

            max_depth_val = self.spin_max_depth.value()
            clf_depth = None if max_depth_val == 0 else max_depth_val
            min_leaf_val = self.spin_min_leaf.value()

            clf = RandomForestClassifier(n_estimators=150, max_depth=clf_depth, min_samples_leaf=min_leaf_val,
                                         n_jobs=-1, random_state=42)
            clf.fit(X_train, y_train, sample_weight=w_train)

            s1, s2, s3 = self.spin_scale_1.value(), self.spin_scale_2.value(), self.spin_scale_3.value()
            new_model_path = os.path.join(self.model_dir, f"HydroBound-ML_v{new_v}.joblib")
            joblib.dump({'clf': clf, 'scales': [s1, s2, s3]}, new_model_path)

            self.current_model_version = new_v
            self.version_label.setText(f"Model Version: v{new_v}")

            self.spin_scale_1.setEnabled(False)
            self.spin_scale_2.setEnabled(False)
            self.spin_scale_3.setEnabled(False)
            self.btn_unlock.setEnabled(True)

            preds_eval = clf.predict(X_eval)
            precision, recall, f1, _ = precision_recall_fscore_support(y_eval, preds_eval, average='binary',
                                                                       zero_division=0)
            iou = jaccard_score(y_eval, preds_eval, zero_division=0)

            acc_manual = np.mean(clf.predict(X_m_val) == y_m_val) if len(X_m_val) > 0 else 0.0
            cm = confusion_matrix(y_eval, preds_eval)
            tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)

            log_path = os.path.join(self.model_dir, "HydroBound-ML_metrics_log.txt")
            with open(log_path, "a") as f:
                f.write(
                    f"--- Training Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Version: v{new_v} ---\n")
                if self.project_metadata:
                    f.write(
                        f"Project Metadata -> Area: {self.project_metadata.get('project_info', {}).get('area_id', 'Unknown')}\n")
                f.write(
                    f"Core-Set Filter: Water >= {w_min}, Land <= {l_max} (Otsu base reduced to {filtered_base_len} before limit)\n")
                f.write(f"Purged {deleted_points} Otsu points.\n")
                f.write(
                    f"Manual Samples: {len(X_manual_all)} (Clones: x{repeats}, Noise: {self.spin_noise_aug.value()}, Weight: x{self.spin_weight.value()})\n")
                f.write(f"Model Gen: Max Depth: {max_depth_val}, Min Leaf: {min_leaf_val}\n")
                f.write(
                    f"Validation Metrics (Global) -> Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | IoU: {iou:.4f}\n")
                f.write(f"Validation Metrics (Hard Manual Cases) -> Accuracy: {acc_manual:.4f}\n")
                f.write(f"Confusion Matrix (TN, FP, FN, TP): {tn}, {fp}, {fn}, {tp}\n\n")

            msg = (f"Model Iteration {new_v} successfully updated!\n\n"
                   f"Purged {deleted_points} conflicting background points.\n"
                   f"Manual points augmented: {len(X_manual_all)} -> {len(X_manual_all) * repeats} variants\n"
                   f"Global F1-Score: {f1:.2f} | Manual Acc: {acc_manual:.2f}\n\n"
                   f"Adjust 'Decision Threshold' if needed, then click 'Predict' to apply.")
            self.clear_manual_samples_ui()
            QMessageBox.information(self, "Success", msg)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Training failed: {e}")
        finally:
            p.close()

    def predict_mask(self):
        src_ndwi = self.get_layer_src('ndwi')
        src_ndvi = self.get_layer_src('ndvi')
        src_ngrdi = self.get_layer_src('ngrdi')

        if not src_ndwi or not src_ndvi or not src_ngrdi:
            QMessageBox.warning(self, "Error", "NDWI, NDVI and NGRDI layers required.")
            return

        v = self.current_model_version
        potential_model_path = os.path.join(self.model_dir, f"HydroBound-ML_v{v}.joblib")
        model_exists = v > 0 and os.path.exists(potential_model_path)
        model_path = potential_model_path if model_exists else ""
        otsu_cache_path = os.path.join(self.model_dir, "HydroBound-ML_otsu_samples_v1.0.0.joblib")

        p = QProgressDialog(f"Predicting ML v{v}..." if model_exists else "Bootstrapping Spatial Otsu...", "Cancel", 0,
                            100, self)
        p.setWindowModality(Qt.WindowModality.ApplicationModal)
        p.setFixedSize(500, 150)
        p.show()
        QCoreApplication.processEvents()

        try:
            p.setLabelText("Reading Global Rasters & Filtering KML AOI...")
            QCoreApplication.processEvents()

            kml_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.kml')]
            aoi_window = None
            aoi_transform = src_ndwi.transform
            valid_conditions = None

            if self.selected_kml_path and os.path.exists(self.selected_kml_path):
                try:
                    aoi_gdf = gpd.read_file(self.selected_kml_path, driver='KML')
                    if aoi_gdf.crs is None:
                        aoi_gdf.set_crs(epsg=4326, inplace=True)
                    if aoi_gdf.crs != src_ndwi.crs:
                        aoi_gdf = aoi_gdf.to_crs(src_ndwi.crs)

                    if not aoi_gdf.empty:
                        minx, miny, maxx, maxy = aoi_gdf.total_bounds
                        aoi_window = from_bounds(minx, miny, maxx, maxy, src_ndwi.transform)
                        aoi_window = aoi_window.intersection(
                            rasterio.windows.Window(0, 0, src_ndwi.width, src_ndwi.height))
                        aoi_window = aoi_window.round_offsets().round_lengths()
                        aoi_transform = rasterio.windows.transform(aoi_window, src_ndwi.transform)
                except Exception as e:
                    pass

            if aoi_window is None:
                aoi_window = rasterio.windows.Window(0, 0, src_ndwi.width, src_ndwi.height)

            def get_arr_cropped(src, win):
                r = src.read(1, window=win)
                if src.dtypes[0] == 'int16':
                    r = r.astype('float32') / 10000.0
                    if src.nodata is not None:
                        r = np.where(np.isclose(r, src.nodata / 10000.0), np.nan, r)
                else:
                    if src.nodata is not None:
                        r = np.where(np.isclose(r, src.nodata), np.nan, r)
                return r

            ndwi = get_arr_cropped(src_ndwi, aoi_window)
            ndvi = get_arr_cropped(src_ndvi, aoi_window)
            ngrdi = get_arr_cropped(src_ngrdi, aoi_window)

            valid_conditions = np.isfinite(ndwi) & np.isfinite(ndvi) & np.isfinite(ngrdi) & (ndwi > -1.5) & (
                        ndvi > -1.5) & (ngrdi > -1.5)

            if self.selected_kml_path and 'aoi_gdf' in locals() and not aoi_gdf.empty:
                aoi_geoms = aoi_gdf.geometry.tolist()
                kml_mask = rasterio.features.geometry_mask(aoi_geoms, out_shape=ndwi.shape, transform=aoi_transform,
                                                           invert=True)
                valid_conditions = valid_conditions & kml_mask

            p.setLabelText("Checking for cached Multi-Scale Textures (14D)...")
            p.setValue(15)
            QCoreApplication.processEvents()

            res_x = abs(aoi_transform[0])
            s1, s2, s3 = self.spin_scale_1.value(), self.spin_scale_2.value(), self.spin_scale_3.value()
            if model_exists:
                try:
                    saved_data = joblib.load(model_path)
                    if isinstance(saved_data, dict) and 'scales' in saved_data:
                        s1, s2, s3 = saved_data['scales']
                except Exception:
                    pass

            scales = [s1, s2, s3]

            kml_id = os.path.splitext(os.path.basename(self.selected_kml_path))[0] if self.selected_kml_path else "full"
            textures_cache_path = os.path.join(self.model_dir,
                                               f"HydroBound-ML_textures_cache_v1.0.0_{kml_id}_{s1}_{s2}_{s3}.joblib")

            recalc = True
            if os.path.exists(textures_cache_path):
                try:
                    scale_features = joblib.load(textures_cache_path)
                    if len(scale_features) > 0 and scale_features[0].shape == ndwi.shape:
                        recalc = False
                except:
                    pass

            if recalc:
                p.setLabelText("Calculating Spatial Features & Gradients...")
                QCoreApplication.processEvents()
                scale_features = []

                for idx, m_scale in enumerate(scales):
                    s_px = max(1, int(round(m_scale / res_x)))
                    if s_px % 2 == 0:
                        s_px += 1

                    for arr in [ndwi, ndvi, ngrdi]:
                        c1 = ndimage.uniform_filter(arr, size=s_px, mode='reflect')
                        c2 = ndimage.uniform_filter(arr ** 2, size=s_px, mode='reflect')
                        feature = np.sqrt(np.clip(c2 - c1 ** 2, 0, None)).astype('float32')
                        scale_features.append(feature)
                        del c1, c2
                        gc.collect()

                    if idx < 2:
                        sig = max(1.0, s_px / 3.0)
                        grad = ndimage.gaussian_gradient_magnitude(np.nan_to_num(ndwi, 0), sigma=sig).astype('float32')
                        scale_features.append(grad)

                    p.setValue(15 + int(20 * ((idx + 1) / 3)))
                    QCoreApplication.processEvents()

                joblib.dump(scale_features, textures_cache_path)
            else:
                p.setLabelText("Loading cached Multi-Scale Textures...")
                QCoreApplication.processEvents()

            water_mask = np.zeros_like(ndwi, dtype='uint8')

            msg_success = ""

            if model_exists:
                p.setLabelText("Executing Inference...")
                p.setValue(40)
                QCoreApplication.processEvents()

                saved_data = joblib.load(model_path)
                clf = saved_data['clf'] if isinstance(saved_data, dict) else saved_data

                valid_indices = np.where(valid_conditions)
                total_valid = len(valid_indices[0])

                prob_map = np.zeros_like(ndwi, dtype='float32')
                water_class_idx = np.where(clf.classes_ == 1)[0][0]

                chunk_size = 1000000
                for start_idx in range(0, total_valid, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_valid)
                    chunk_rows, chunk_cols = valid_indices[0][start_idx:end_idx], valid_indices[1][start_idx:end_idx]

                    X_chunk = np.column_stack((
                        ndwi[chunk_rows, chunk_cols], ndvi[chunk_rows, chunk_cols], ngrdi[chunk_rows, chunk_cols],
                        scale_features[0][chunk_rows, chunk_cols], scale_features[1][chunk_rows, chunk_cols],
                        scale_features[2][chunk_rows, chunk_cols], scale_features[3][chunk_rows, chunk_cols],
                        scale_features[4][chunk_rows, chunk_cols], scale_features[5][chunk_rows, chunk_cols],
                        scale_features[6][chunk_rows, chunk_cols], scale_features[7][chunk_rows, chunk_cols],
                        scale_features[8][chunk_rows, chunk_cols], scale_features[9][chunk_rows, chunk_cols],
                        scale_features[10][chunk_rows, chunk_cols]
                    ))

                    y_proba = clf.predict_proba(X_chunk)[:, water_class_idx]
                    prob_map[chunk_rows, chunk_cols] = y_proba
                    del X_chunk
                    gc.collect()

                    p.setValue(40 + int(30 * (end_idx / total_valid)))
                    QCoreApplication.processEvents()

                p.setLabelText("Binarizing ML Output via Strict Threshold...")
                p.setValue(75)
                QCoreApplication.processEvents()

                thresh = self.spin_prob_thresh.value()
                water_mask = (prob_map >= thresh).astype('uint8')
                water_mask[~valid_conditions] = 0

                msg_success += f"Global ML mask generated with strict threshold: {thresh:.2f}!\n"
                del prob_map
                gc.collect()

            else:
                p.setLabelText("Calculating Spatial Otsu Base...")
                p.setValue(40)
                QCoreApplication.processEvents()
                
                from skimage.filters import threshold_otsu
                valid_ndwi = ndwi[valid_conditions]
                
                if len(valid_ndwi) > 0:
                    try:
                        t_otsu = threshold_otsu(valid_ndwi)
                    except Exception:
                        t_otsu = 0.0
                else:
                    t_otsu = 0.0
                    
                water_mask = np.zeros_like(ndwi, dtype='uint8')
                water_mask[valid_conditions & (ndwi >= t_otsu)] = 1
                
                p.setLabelText("Bootstrapping Spatial ML Base...")
                p.setValue(55)
                QCoreApplication.processEvents()
                
                water_idx = np.where((valid_conditions) & (water_mask == 1))
                land_idx = np.where((valid_conditions) & (water_mask == 0))
                
                limit = 10000 
                w_sel = np.random.choice(len(water_idx[0]), min(len(water_idx[0]), limit), replace=False)
                l_sel = np.random.choice(len(land_idx[0]), min(len(land_idx[0]), limit), replace=False)
                
                w_rows, w_cols = water_idx[0][w_sel], water_idx[1][w_sel]
                l_rows, l_cols = land_idx[0][l_sel], land_idx[1][l_sel]
                
                coords_w_x = aoi_transform[2] + w_cols * aoi_transform[0] + w_rows * aoi_transform[1]
                coords_w_y = aoi_transform[5] + w_cols * aoi_transform[3] + w_rows * aoi_transform[4]
                coords_l_x = aoi_transform[2] + l_cols * aoi_transform[0] + l_rows * aoi_transform[1]
                coords_l_y = aoi_transform[5] + l_cols * aoi_transform[3] + l_rows * aoi_transform[4]
                
                X_w = np.column_stack((
                    ndwi[w_rows, w_cols], ndvi[w_rows, w_cols], ngrdi[w_rows, w_cols], 
                    scale_features[0][w_rows, w_cols], scale_features[1][w_rows, w_cols], scale_features[2][w_rows, w_cols], scale_features[3][w_rows, w_cols],
                    scale_features[4][w_rows, w_cols], scale_features[5][w_rows, w_cols], scale_features[6][w_rows, w_cols], scale_features[7][w_rows, w_cols],
                    scale_features[8][w_rows, w_cols], scale_features[9][w_rows, w_cols], scale_features[10][w_rows, w_cols]
                ))
                X_l = np.column_stack((
                    ndwi[l_rows, l_cols], ndvi[l_rows, l_cols], ngrdi[l_rows, l_cols], 
                    scale_features[0][l_rows, l_cols], scale_features[1][l_rows, l_cols], scale_features[2][l_rows, l_cols], scale_features[3][l_rows, l_cols],
                    scale_features[4][l_rows, l_cols], scale_features[5][l_rows, l_cols], scale_features[6][l_rows, l_cols], scale_features[7][l_rows, l_cols],
                    scale_features[8][l_rows, l_cols], scale_features[9][l_rows, l_cols], scale_features[10][l_rows, l_cols]
                ))
                
                X_train_base = np.vstack((X_w, X_l))
                y_train_base = np.concatenate((np.ones(len(X_w)), np.zeros(len(X_l))))
                coords_base = np.vstack((np.column_stack((coords_w_x, coords_w_y)), np.column_stack((coords_l_x, coords_l_y))))
                
                joblib.dump({'X': X_train_base, 'y': y_train_base, 'coords': coords_base, 'version': '1.0.7'}, otsu_cache_path)
                msg_success += f"Spatial Otsu Base generated and cached! (Threshold: {t_otsu:.2f})\n"
                
            del scale_features
            gc.collect()

            if self.check_vector_snap.isChecked():
                p.setLabelText("Generating Superpixels (OBIA)...")
                QCoreApplication.processEvents()

                img_slic = np.dstack((
                    np.nan_to_num(ndwi, nan=0.0),
                    np.nan_to_num(ndvi, nan=0.0),
                    np.nan_to_num(ngrdi, nan=0.0)
                ))

                img_slic = (img_slic + 1.0) / 2.0

                target_area_m2 = self.spin_slic_area.value()
                compactness = self.spin_slic_compactness.value()
                pixel_area_m2 = res_x * abs(aoi_transform[4])
                pixels_per_segment = target_area_m2 / pixel_area_m2
                n_segments = max(10, int((img_slic.shape[0] * img_slic.shape[1]) / pixels_per_segment))

                segments = slic(img_slic, n_segments=n_segments, compactness=compactness, start_label=1)

                p.setLabelText("Fusing Geometry (OBIA) with Semantics (ML)...")
                QCoreApplication.processEvents()

                seg_flat = segments.ravel()
                water_flat = water_mask.ravel()

                seg_counts = np.bincount(seg_flat)
                water_counts = np.bincount(seg_flat, weights=water_flat)

                with np.errstate(divide='ignore', invalid='ignore'):
                    water_ratio = np.where(seg_counts > 0, water_counts / seg_counts, 0)

                obia_ratio_thresh = self.spin_obia_ratio.value()
                obia_water_mask = (water_ratio[segments] >= obia_ratio_thresh).astype('uint8')

                water_mask = obia_water_mask
                msg_success += f"OBIA Superpixel Snap applied (Ratio: {obia_ratio_thresh}).\n"

            del ndwi, ndvi, ngrdi
            gc.collect()

            p.setLabelText("Patching micro-holes (Morphological Closing)...")
            QCoreApplication.processEvents()
            water_mask = ndimage.binary_closing(water_mask, structure=np.ones((3, 3))).astype('uint8')

            p.setLabelText("Vectorizing final mask...")
            p.setValue(85)
            QCoreApplication.processEvents()

            shapes = rasterio.features.shapes(water_mask, mask=(water_mask == 1), transform=aoi_transform)
            polygons = [shape(geom) for geom, value in shapes]
            gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=src_ndwi.crs)
            del water_mask
            gc.collect()

            if not gdf.empty:
                noise_threshold_pct = self.spin_noise_filter.value() / 100.0
                if noise_threshold_pct > 0.0:
                    p.setLabelText("Filtering Island Noise...")
                    p.setValue(90)
                    QCoreApplication.processEvents()
                    max_area = gdf.geometry.area.max()
                    gdf = gdf[gdf.geometry.area >= max_area * noise_threshold_pct]

                smooth_radius = self.spin_smooth_radius.value()
                if smooth_radius > 0.0:
                    p.setLabelText("Applying Organic Smoothing...")
                    QCoreApplication.processEvents()
                    gdf['geometry'] = gdf.geometry.buffer(smooth_radius, join_style=1).buffer(-smooth_radius * 2,
                                                                                              join_style=1).buffer(
                        smooth_radius, join_style=1)

                gdf['geometry'] = gdf.geometry.simplify(tolerance=res_x * 0.5, preserve_topology=False)
                gdf['geometry'] = gdf.geometry.buffer(0)
                gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]

                self.vector_mask = gdf
                self._cache_vector_lines(self.vector_mask)
                self.check_mask.setChecked(True)
                self.toggle_vector_visibility()

                self.edit_mode = None
                for b in self.edit_btns: b.setChecked(False)
                self.btn_measure.setChecked(False)
                self.clear_edit_drawing()
                self.clear_measurements()

                QMessageBox.information(self, "Success", msg_success)
            else:
                QMessageBox.warning(self, "Warning", "No water boundary detected.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {e}")
        finally:
            p.close()

    def export_geojson(self):
        if self.vector_mask is None or self.vector_mask.empty:
            QMessageBox.warning(self, "Error", "No vector to export. Run 'Predict' first.")
            return

        epsg_text = self.line_epsg.text().strip()
        rfc7946_val = 'YES' if self.combo_rfc.currentText() == 'Yes' else 'NO'
        coord_prec = self.spin_coord_precision.value()

        try:
            target_crs = pyproj.CRS.from_user_input(epsg_text)
        except Exception as e:
            QMessageBox.critical(self, "CRS Error",
                                 f"Invalid EPSG code: '{epsg_text}'.\nPlease provide a valid code (e.g., EPSG:4979).\n\nDetails: {e}")
            return

        p = QProgressDialog("Exporting GeoJSON...", "Cancel", 0, 100, self)
        p.setWindowModality(Qt.WindowModality.ApplicationModal)
        p.show()
        QCoreApplication.processEvents()

        try:
            timestamp = time.strftime("%y%m%d%H%M")
            p.setLabelText("Exporting standard polygon...")
            p.setValue(20)
            QCoreApplication.processEvents()

            export_poly = self.vector_mask.copy()
            export_poly = export_poly.explode(index_parts=False).reset_index(drop=True)
            export_poly = export_poly[export_poly.geometry.type == 'Polygon']

            dtm_src = self.get_layer_src("dtm")
            z_method = self.combo_z_method.currentText() if self.combo_z_method.currentText() else 'mean'
            
            if dtm_src:
                p.setLabelText("Calculating Altitudes for standard polygon...")
                QCoreApplication.processEvents()
                altitudes_base = []
                for geom in export_poly.geometry:
                    try:
                        out_image, _ = rasterio.mask.mask(dtm_src, [geom], crop=True, filled=False)
                        valid_data = out_image.compressed()
                        if dtm_src.nodata is not None:
                            valid_data = valid_data[valid_data != dtm_src.nodata]
                        if len(valid_data) > 0:
                            val = np.mean(valid_data) if z_method == 'mean' else np.min(valid_data) if z_method == 'min' else np.max(valid_data)
                            altitudes_base.append(round(float(val), 2))
                        else:
                            altitudes_base.append(np.nan)
                    except Exception:
                        altitudes_base.append(np.nan)
                export_poly['altitude'] = altitudes_base
            else:
                export_poly['altitude'] = np.nan

            export_poly = export_poly.to_crs(target_crs)                
            export_poly = export_poly[['geometry', 'altitude']]

            out_nogrid = os.path.join(self.output_dir, f"{timestamp}_water_boundary_polygon.geojson")
            export_poly.to_file(out_nogrid, driver='GeoJSON', engine='fiona', RFC7946=rfc7946_val, COORDINATE_PRECISION=coord_prec)

            if self.check_export_grid.isChecked():
                p.setLabelText("Generating grid...")
                p.setValue(40)
                QCoreApplication.processEvents()
                grid_size = self.spin_grid_size.value()

                minx, miny, maxx, maxy = self.vector_mask.total_bounds
                x_coords = np.arange(minx, maxx, grid_size)
                y_coords = np.arange(miny, maxy, grid_size)

                grid_polys = [box(x, y, x + grid_size, y + grid_size) for x in x_coords for y in y_coords]
                grid_gdf = gpd.GeoDataFrame({'geometry': grid_polys}, crs=self.vector_mask.crs)

                p.setLabelText("Clipping grid...")
                p.setValue(60)
                QCoreApplication.processEvents()
                clipped_grid = gpd.overlay(grid_gdf, self.vector_mask, how='intersection')
                clipped_grid = clipped_grid[clipped_grid.geometry.type.isin(['Polygon', 'MultiPolygon'])]

                p.setLabelText("Calculating Altitudes for grid...")
                p.setValue(80)
                QCoreApplication.processEvents()
                if dtm_src:
                    altitudes = []
                    for geom in clipped_grid.geometry:
                        try:
                            out_image, _ = rasterio.mask.mask(dtm_src, [geom], crop=True, filled=False)
                            valid_data = out_image.compressed()
                            if dtm_src.nodata is not None:
                                valid_data = valid_data[valid_data != dtm_src.nodata]
                            if len(valid_data) > 0:
                                val = np.mean(valid_data) if z_method == 'mean' else np.min(
                                    valid_data) if z_method == 'min' else np.max(valid_data)
                                altitudes.append(round(float(val), 2))
                            else:
                                altitudes.append(np.nan)
                        except Exception:
                            altitudes.append(np.nan)
                    clipped_grid['altitude'] = altitudes
                else:
                    clipped_grid['altitude'] = np.nan

                p.setLabelText("Structuring Topology...")
                QCoreApplication.processEvents()

                clipped_grid = clipped_grid.to_crs(target_crs)
                clipped_grid = clipped_grid.explode(index_parts=False).reset_index(drop=True)
                clipped_grid = clipped_grid[clipped_grid.geometry.type == 'Polygon']

                clipped_grid = clipped_grid[['geometry', 'altitude']]

                p.setLabelText("Exporting...")
                p.setValue(95)
                QCoreApplication.processEvents()

                out_grid = os.path.join(self.output_dir, f"{timestamp}_water_boundary_polygon_grid.geojson")
                clipped_grid.to_file(out_grid, driver='GeoJSON', engine='fiona', RFC7946=rfc7946_val, COORDINATE_PRECISION=coord_prec)
                QMessageBox.information(self, "Success", f"Saved:\n1) {out_nogrid}\n2) {out_grid}")
            else:
                p.setValue(100)
                QMessageBox.information(self, "Success", f"Saved standard polygon:\n{out_nogrid}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")
        finally:
            p.close()


if __name__ == "__main__":
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    ex = HydroBound_ML_App()
    ex.show()
    sys.exit(app.exec())