# HydroBound-ML Assistant (v.1.0.7)

An open-source, advanced hybrid machine learning and Object-Based Image Analysis (OBIA) tool designed for automated water body delineation and high-precision boundary vectorization from aerial and UAV-based multispectral imagery.

## 📖 Scientific Publication & Citation
This software, along with the accompanying dataset and trained models, is an integral part of the research cycle developed for the doctoral dissertation at the Military University of Technology (Warsaw, Poland). If you use this software in your research, please cite:

> **Wróblewski, P., & Fryśkowska-Skibniewska, A. (2026).** *HydroBound-ML: A Hybrid Machine Learning and Object-Based Image Analysis Approach for Automated Water Boundary Delineation in Airborne Lidar Bathymetry*. Sensors. 
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
* **Automated Data Processing:** Generation of continuous water masks with robust 3D DTM attribution (ellipsoidal height extraction) suitable for Exponential Decomposition software (e.g., RiPROCESS).

---

## 📦 Dataset & Pre-trained Models
Due to GitHub's file size constraints, the complete project archive—including the `Magnuszew_2026` training dataset, the pre-trained Machine Learning models (`.joblib`), corresponding performance metrics, full project directory structure, and auxiliary validation data—is hosted externally on the FigShare repository.

🔗 **Access the full dataset and models here:** [https://doi.org/10.6084/m9.figshare.33366177](https://doi.org/10.6084/m9.figshare.33366177)

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
 │    ├── dtm.tif                    # Optional (required for 3D elevation attribution)
 │    ├── boundary.kml               # Optional (AOI masking)
 │    └── metadata.json              # Required (project context tracking)
 ├── 📂 OUTPUT/                      # Result directory
 │    ├── [timestamp]_water_boundary_polygon.geojson       # Primary vector mask (with 3D altitude attribute)
 │    └── [timestamp]_water_boundary_polygon_grid.geojson  # Optional 3D-attributed grid logic
 └── 📂 MODEL/                       # Machine Learning artifacts
      ├── HydroBound-ML_vX.joblib             # Compiled Random Forest model
      ├── HydroBound-ML_otsu_samples...       # Background bootstrapped samples
      ├── HydroBound-ML_vX_manual_samples...  # User-defined training points
      ├── HydroBound-ML_textures_cache...     # 14D Feature space cache
      └── HydroBound-ML_metrics_log.txt       # Training performance logs
```

Data Dependency Logic
\* Indices (NDWI, NDVI, NGRDI): The application first checks for the presence of these files in the INPUT folder. If all three are present, they are loaded directly. If any index is missing, the application requires both ortho_rgb.tif and ortho_nir.tif to automatically generate the missing files.

\* Ortho Imagery: While ortho_rgb and ortho_nir are essential for generating indices, they are not strictly required if the indices are already pre-computed and present in the INPUT folder. However, it is recommended they remain in the workspace for traceability.

\* Model Features: The machine learning model relies solely on the spectral indices and their derived spatial textures. Once indices are available, source orthomosaics are not invoked during training or prediction phases.

📊 Pre-trained Model Performance (v1.0.7)
The included robust Random Forest model was trained on the Magnuszew_2026 dataset (available publicly on FigShare) to combat extreme infrastructural noise (bridges, railways, dark asphalt) overlapping with turbid river waters.

Data Integrity Architecture: To ensure strict scientific validation and prevent spatial information leakage during the Human-in-the-Loop active learning phase, the framework incorporates a density-based spatial clustering algorithm (DBSCAN) combined with a Group Shuffle Split (80:20). This guarantees that jitter-augmented points adjacent to a single manual click strictly fall into the same validation subset.

Manual Samples: 176,598 (Bootstrapped and Augmented)

Global Precision: 0.908 | Global Recall: 0.941

F1-Score: 0.924 | IoU (Jaccard Index): 0.859

(Note: These represent strict, non-augmented internal validation metrics. External benchmark testing yielded an independent F1-Score of 0.95).

🛡️ Installation & Requirements
Requires Python 3.10+. Use a dedicated virtual environment (venv):

```
Bash
python -m venv venv
venv\Scripts\activate
pip install numpy scipy scikit-learn scikit-image rasterio geopandas shapely pyproj fiona joblib matplotlib pyqt6 jupyter ipykernel
```

Execute via terminal:

```
Bash
python "HydroBound-ML Assistant v.1.0.7 (Release).py"
```

License: MIT License (See LICENSE file for details).
