# Adapting SAM2-UNet for High-Resolution LULC Semantic Segmentation with OpenStreetMap Auxiliary Supervision

This repository will contain the official PyTorch implementation for the paper "Adapting SAM2-UNet for High-Resolution LULC Semantic Segmentation with OpenStreetMap Auxiliary Supervision".

## Overview
Vision foundation models offer powerful representations for Earth surface segmentation, but thin and topologically complex classes (such as roads and buildings) remain challenging. 

Our framework adapts the SAM2-UNet architecture for high-resolution Land Use and Land Cover (LULC) mapping. We utilize OpenStreetMap (OSM) vector data as a privileged, training-time-only auxiliary signal to improve boundary quality and structural continuity. At inference time, the auxiliary branch is discarded, allowing the model to operate entirely on standard RGB imagery without requiring external geospatial inputs.

## Code Availability
**Note:** The codebase is currently being cleaned and prepared for public release. The full repository, including the model architecture, distance-transform loss implementations, data preprocessing pipelines, and evaluation notebooks, will be uploaded here soon.
