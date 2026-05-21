# HydroBound-ML Assistant (v.1.0.0)

An open-source, advanced hybrid machine learning and Object-Based Image Analysis (OBIA) tool designed for automated water body delineation and high-precision boundary vectorization from aerial and UAV-based multispectral imagery.

## 📖 Scientific Publication & Citation
This software, along with the accompanying dataset and trained models, is an integral part of the research cycle developed for the doctoral dissertation at the Military University of Technology (Warsaw, Poland). If you use this software in your research, please cite:

> **Wróblewski, P. (2026).** *[Insert Full Title of the Paper Here]*. [Insert Journal Name]. 
> **DOI:** [Insert Link/DOI Here when published]

### 👨‍💻 Author & Affiliation
**Patryk Wróblewski**
* **[EN]** Department of Imagery Intelligence, Faculty of Civil Engineering and Geodesy, Military University of Technology, 2 gen. Sylwestra Kaliskiego St., 00-908 Warsaw, Poland.
* **[PL]** Katedra Rozpoznania Obrazowego, Wydział Inżynierii Lądowej i Geodezji, Wojskowa Akademia Techniczna im. Jarosława Dąbrowskiego, ul. gen. Sylwestra Kaliskiego 2, 00-908 Warszawa.

---

## 🎯 Purpose and Applications
HydroBound-ML is a versatile GIS tool developed to resolve spatial ambiguities and optimize water surface definition in demanding Airborne Lidar Bathymetry (ALB) workflows. It excels in:
* Updating and harmonizing high-resolution topographic databases (e.g., BDOT10k, BDOO).
* Floodplain mapping and multi-sensor hydrographic vectorization.
* **Semantic-Geometric Decoupling:** Utilizing ML for pixel-level semantic meaning (14D feature space) and OBIA (SLIC superpixels) for crisp, physically accurate geometry.

---

## 📁 Project Structure

The software expects a structured workspace. While the program can auto-generate spectral indices from source orthomosaics, it also accepts pre-computed indices if provided.

```text
📂 Project_Workspace/
 ├── 📂 INPUT/                       # Data repository
 │    ├── ortho_rgb.tif              # Source 1 (Required if indices are missing)
 │    ├── ortho_nir.tif              # Source 2 (Required if indices are missing)
 │    ├── ndwi.tif                   # Optional (pre-computed or auto-generated)
 │    ├── ndvi.tif                   # Optional (pre-computed or auto-generated)
 │    ├── ngrdi.tif                  # Optional (pre-computed or auto-generated)
 │    ├── dtm.tif                    # Optional (required for 3D grid export)
 │    ├── boundary.kml               # Optional (AOI masking)
 │    └── metadata.json              # Required (project context tracking)
 ├── 📂 OUTPUT/                      # Result directory
 │    ├── [timestamp]_water_boundary.geojson      # Primary vector mask
 │    └── [timestamp]_water_boundary_grid.geojson # Optional 3D-attributed grid
 └── 📂 MODEL/                       # Machine Learning artifacts
      ├── HydroBound-ML_v1.joblib             # Compiled Random Forest model
      ├── HydroBound-ML_otsu_samples.joblib   # Background bootstrapped samples
      ├── HydroBound-ML_manual_samples.joblib # User-defined training points
      ├── HydroBound-ML_textures_cache.joblib # 14D Feature space cache
      └── HydroBound-ML_metrics_log.txt       # Training performance logs
Data Dependency Logic
Indices (NDWI, NDVI, NGRDI): The application first checks for the presence of these files in the INPUT folder. If all three are present, they are loaded directly. If any index is missing, the application requires both ortho_rgb.tif and ortho_nir.tif to automatically generate the missing files.

Ortho Imagery: While ortho_rgb and ortho_nir are essential for generating indices, they are not strictly required if the indices are already pre-computed and present in the INPUT folder. However, it is recommended they remain in the workspace for traceability.

Model Features: The machine learning model relies solely on the spectral indices and their derived spatial textures. Once indices are available, source orthomosaics are not invoked during training or prediction phases.

📊 Pre-trained Model Performance (v1)
The model was trained on the Magnuszew_2026 dataset to combat extreme infrastructural noise (bridges, railways, dark asphalt) overlapping with turbid river waters.

Manual Samples: 176,598 (Augmented)

Global Precision: 0.9778 | Global Recall: 0.9732

F1-Score: 0.9755 | IoU (Jaccard Index): 0.9522

🛡️ Installation & Requirements
Requires Python 3.10+. Use a dedicated virtual environment (venv):

Bash
python -m venv venv
venv\Scripts\activate
pip install numpy scipy scikit-learn scikit-image rasterio geopandas shapely pyproj fiona joblib matplotlib pyqt6 jupyter ipykernel
Execute via terminal: python "HydroBound-ML Assistant v.1.0.0.py"

---
**License:** MIT License (See `LICENSE` file for details).