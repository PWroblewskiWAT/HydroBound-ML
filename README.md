# HydroBound-ML Assistant (v.1.0.0)

An open-source, advanced hybrid machine learning and Object-Based Image Analysis (OBIA) tool designed for automated water body delineation and high-precision boundary vectorization from aerial and UAV-based multispectral imagery.

## 📖 Scientific Publication & Citation
This software is an integral part of the research cycle developed for the doctoral dissertation at the Military University of Technology (Warsaw, Poland). If you use this software, model, or dataset in your research, please cite our paper:

> **Wróblewski, P. (2026).** *[Insert Full Title of the Paper Here]*. [Insert Journal Name]. 
> **DOI:** [Insert Link/DOI Here when published]

### 👨‍💻 Author & Affiliation
**Patryk Wróblewski**
* **[EN]** Department of Imagery Intelligence, Faculty of Civil Engineering and Geodesy, Military University of Technology, 2 gen. Sylwestra Kaliskiego St., 00-908 Warsaw, Poland.
* **[PL]** Katedra Rozpoznania Obrazowego, Wydział Inżynierii Lądowej i Geodezji, Wojskowa Akademia Techniczna im. Jarosława Dąbrowskiego, ul. gen. Sylwestra Kaliskiego 2, 00-908 Warszawa.

---

## 🎯 Purpose and Applications
While originally engineered to resolve spatial ambiguities and optimize water surface definition in demanding **Airborne Lidar Bathymetry (ALB)** workflows (e.g., target extraction via Exponential Decomposition), HydroBound-ML is a highly versatile GIS tool. It excels in:
* Updating and harmonizing high-resolution topographic databases (e.g., BDOT10k, BDOO).
* Floodplain mapping and hydrographic vectorization.
* Semantic-Geometric Decoupling: Utilizing ML for pixel-level semantic meaning, and OBIA (SLIC superpixels) for crisp, physically accurate geometry.

---

## 🚀 Core Pipeline & Features

### Step 1: Project Initialization & Automated Indexing
* **Directory Structure:** The software strictly requires an `INPUT` directory containing your source rasters (`.tif`) and an optional AOI boundary (`.kml`).
* **Required Data:** RGB Orthomosaic (`ortho_rgb.tif`) and NIR Orthomosaic (`ortho_nir.tif`). Digital Terrain Model (`dtm.tif`) is optional but required for 3D Grid Export.
* **Smart Indexing:** If NDWI, NDVI, or NGRDI are missing, the software dynamically generates them. It queries the user to map the correct spectral bands from multi-band inputs, utilizing **GDAL WarpedVRT** and **Macro-Chunking (4096px)** for extremely fast, memory-efficient spatial realignment and execution. 
* **Overviews (Pyramids):** Generated rasters are automatically optimized with Cloud Optimized GeoTIFF (COG) principles (internal overviews built via nearest neighbor resampling) ensuring lightning-fast GUI rendering without CPU bottlenecks.

### Step 2: Prediction & Post-Processing (OBIA)
* **Semantic Inference:** A trained Random Forest classifier infers water probability based on a multi-scale 14D spatial-spectral feature space (pixel values, standard deviations, Gaussian Gradient Magnitudes).
* **SLIC Superpixel Snap (OBIA):** Overcomes the "jagged edge" limitation of standard ML models. It dynamically clusters pixels into homogeneous segments, applying a majority-vote logic (Edge Exp. Ratio) to snap boundaries to physical shorelines.
* **Dynamic LoD Coupling:** Superpixel size and vector smoothing radiuses are mathematically coupled to the ML's Micro-Scale parameter, ensuring logical integrity between semantic perception and geometric output.

### Step 3: Model Training & Updating
* **Spatial Otsu Bootstrapping:** Automatically samples background data using histogram thresholding (Otsu method on NDWI). 
* **Interactive Manual Sampling:** Left/Right click on the canvas to inject user-defined Water/Land samples.
* **Data Augmentation:** Features jittering (spatial radius offsets) and signal noise injection to robustly clone user samples, preventing model overfitting while purging conflicting Otsu errors (using cKDTree).

### Step 3.5: Expert Vector Editing (Spatial Boolean Operations)
The software includes a robust internal GIS editor for topologic correction before export:
* `Delete Polygon Part`: Instantly drops isolated false-positive "island" polygons.
* `Fill Inner Hole`: Patches false-negative gaps inside water bodies.
* `Draw Bridge / Add`: Draw polygons that are mathematically united (`Union`) with the main mask.
* `Draw Cut / Remove`: Draw polygons that are subtracted (`Difference`) from the main mask.

### Step 4: GeoJSON Export & 3D Gridding
* Exports the final, topologically clean multi-polygon into a standardized `GeoJSON`.
* **Grid Export:** Optionally chunks the vector mask into a defined regular grid (e.g., 500m) and intersects it with the DTM to extract statistical altitudes (Mean, Min, Max), appending them to the GeoJSON attributes.

---

## 🛡️ Built-in Safeguards & Error Handling
* **State Leak Prevention (Hard Resets):** Upon loading a new project, the software performs a brutal hard-reset of the GDAL environment (`rasterio.Env`), flushes I/O buffers, and invokes the Python garbage collector (`gc.collect()`). This prevents memory freezing and ensures zero cross-contamination between consecutive project loads.
* **NoData Isolation:** Rigorously tracks background NoData values across multiple source sensors to prevent histogram corruption during array normalization and index calculation.
* **Topology Healing:** User-drawn editing vectors undergo automatic `.buffer(0)` self-healing to prevent fatal Boolean intersection errors.

---

## 📦 Requirements & Installation
This software requires **Python 3.10+** and relies heavily on spatial data science libraries.
Main dependencies include:
* `numpy`, `scipy`, `scikit-learn`, `skimage`
* `rasterio`, `geopandas`, `shapely`, `fiona`, `pyproj`
* `PyQt6`, `matplotlib`

**Note on PROJ Library:**
If you encounter CRS transformation errors related to EPSG codes, the application runs a built-in `fix_proj()` function attempting to map the local PROJ directory automatically. 

---
**License:** MIT License (See `LICENSE` file for details).
