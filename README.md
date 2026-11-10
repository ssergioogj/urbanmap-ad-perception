# UrbanMap: Multi-Camera BEV Perception for Autonomous Driving

[![Status: In Progress](https://img.shields.io/badge/Status-In%20Progress-amber.svg)](https://github.com/ssergioogj/urbanmap-ad-perception)
[![Python 3.12](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end multi-view 3D perception pipeline based on **Lift-Splat-Shoot (LSS)** for autonomous vehicles. The system ingests multi-camera surrounding images and predicts dense, metric Bird's-Eye-View (BEV) semantic cost maps to inform downstream trajectory planning and collision avoidance modules.

---

## Architecture Overview

1. Feature Extraction: Lightweight EfficientNet-B0 backbone extracting 2D multiscale contextual visual features.

2. Lift (Outer Product): Generates per-pixel categorical depth distributions along camera frustums.

3. Splat (Voxel Pooling): Projects discrete 3D frustum points into regular BEV pillars using cumulative sum pooling.

4. BEV Decoder: U-Net convolutional neck processing top-down representations for semantic segmentation (vehicles, lanes, drivable area).

5. Decision Module: Downstream rule-based / fuzzy logic evaluation for tactical lane change and collision risk scoring.
