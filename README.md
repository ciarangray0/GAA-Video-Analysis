# GAA Player Detection Comptuter Vision Pipeline


# GAA Player Detection Computer Vision Pipeline

A 4th year BSc Computer Science project that processes Veo-recorded GAA matches to detect players, track movement, and extract match statistics.

## Overview
This repository implements a computer vision pipeline to:
- Detect players and relevant objects (ball, goals, lines)
- Track player movements across frames
- Aggregate per-player and per-match statistics for analysis

Designed for reproducibility and modular development (detection, tracking, post-processing, analytics).

## Features
- Frame extraction from Veo recordings
- Object detection with configurable models
- Multi-object tracking across frames
- Event and heatmap generation
- Exportable CSV/JSON statistics for downstream analysis

## Pipeline overview
1. Video decoding & frame extraction
2. Frame preprocessing (resize, color normalize)
3. Object detection per frame
4. Tracking (associate detections across frames)
5. Post-processing (filtering, smoothing)
6. Statistics extraction and export (CSV / JSON / visualizations)

Author: ciarangray0 
