"""
GAA Video Analysis Pipeline

This package provides the core video analysis functionality:
- detect: YOLO + ByteTrack player detection and tracking
- homography: Pitch calibration and coordinate mapping
- map_players: Map pixel coordinates to pitch coordinates
- trajectories: Interpolate player trajectories between anchor frames
- schemas: Pydantic models for API request/response validation
- config: Configuration constants
- gaa_pitch_config: GAA pitch vertex definitions
- video: Video metadata utilities
"""
