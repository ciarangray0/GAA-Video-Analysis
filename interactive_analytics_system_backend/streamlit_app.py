"""Streamlit frontend for GAA Video Analysis System."""
import streamlit as st
import requests
import json
from typing import List, Dict, Optional
import cv2
import numpy as np
from PIL import Image
import io

# Backend API URL
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="GAA Video Analysis",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ GAA Video Analysis System")
st.markdown("Upload videos, track players, and analyze trajectories")

# Initialize session state
if 'video_id' not in st.session_state:
    st.session_state.video_id = None
if 'video_metadata' not in st.session_state:
    st.session_state.video_metadata = None
if 'annotations' not in st.session_state:
    st.session_state.annotations = {}  # {frame_idx: [PitchPoint]}
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'current_frame_idx' not in st.session_state:
    st.session_state.current_frame_idx = 0
if 'player_positions' not in st.session_state:
    st.session_state.player_positions = []


def call_api(endpoint: str, method: str = "GET", data: dict = None, files: dict = None, params: dict = None):
    """Helper to call FastAPI backend."""
    url = f"{API_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, params=params)
            else:
                response = requests.post(url, json=data, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure FastAPI server is running on http://localhost:8000")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


# Sidebar for navigation
st.sidebar.title("Pipeline Steps")
step = st.sidebar.radio(
    "Select Step",
    ["1. Upload Video", "2. Run Tracking", "3. Annotate Pitch", "4. Compute Homographies", 
     "5. Map Players", "6. Interpolate", "7. View Results"]
)

# Step 1: Upload Video
if step == "1. Upload Video":
    st.header("Step 1: Upload Video")
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        
        if st.button("Upload to Backend"):
            with st.spinner("Uploading video..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                result = call_api("/videos", method="POST", files=files)
                
                if result:
                    st.session_state.video_id = result["video_id"]
                    st.session_state.video_metadata = {
                        "fps": result["fps"],
                        "num_frames": result["num_frames"]
                    }
                    st.success(f"✅ Video uploaded! ID: {result['video_id']}")
                    st.info(f"FPS: {result['fps']}, Frames: {result['num_frames']}")
    
    if st.session_state.video_id:
        st.success(f"✅ Current Video ID: {st.session_state.video_id}")

# Step 2: Run Tracking
elif step == "2. Run Tracking":
    st.header("Step 2: Run YOLO + ByteTrack Tracking")
    
    if not st.session_state.video_id:
        st.warning("⚠️ Please upload a video first (Step 1)")
    else:
        if st.button("Run Tracking"):
            with st.spinner("Running YOLO + ByteTrack (this may take a while)..."):
                result = call_api(f"/videos/{st.session_state.video_id}/track", method="POST")
                
                if result:
                    st.success(f"✅ Tracking complete!")
                    st.info(f"Frames processed: {result['frames_processed']}, Unique tracks: {result['tracks']}")

# Step 3: Annotate Pitch
elif step == "3. Annotate Pitch":
    st.header("Step 3: Annotate Pitch Points")
    
    if not st.session_state.video_id:
        st.warning("⚠️ Please upload a video first (Step 1)")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col2:
            frame_idx = st.number_input(
                "Frame Index",
                min_value=0,
                max_value=st.session_state.video_metadata["num_frames"] - 1 if st.session_state.video_metadata else 0,
                value=st.session_state.current_frame_idx,
                step=1
            )
            
            if st.button("Load Frame"):
                # Load frame from video file
                video_path = f"data/videos/{st.session_state.video_id}.mp4"
                try:
                    cap = cv2.VideoCapture(video_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.session_state.current_frame = frame_rgb
                        st.session_state.current_frame_idx = frame_idx
                    else:
                        st.error("Failed to load frame")
                except Exception as e:
                    st.error(f"Error loading frame: {str(e)}")
            
            st.markdown("---")
            st.markdown("### Pitch Point Types")
            
            pitch_point_types = {
                "corner_tl": "Top Left Corner",
                "corner_tr": "Top Right Corner",
                "corner_bl": "Bottom Left Corner",
                "corner_br": "Bottom Right Corner",
                "top_goal_lp": "Top Goal Left Post",
                "top_goal_rp": "Top Goal Right Post",
                "bottom_goal_lp": "Bottom Goal Left Post",
                "bottom_goal_rp": "Bottom Goal Right Post",
                "left_box_top": "Left Box Top",
                "right_box_top": "Right Box Top",
                "left_box_bottom": "Left Box Bottom",
                "right_box_bottom": "Right Box Bottom",
            }
            
            selected_pitch_id = st.selectbox(
                "Select Pitch Point Type",
                options=list(pitch_point_types.keys()),
                format_func=lambda x: pitch_point_types[x]
            )
            
            if st.button("Clear Points"):
                if frame_idx in st.session_state.annotations:
                    del st.session_state.annotations[frame_idx]
                st.rerun()
        
        with col1:
            if st.session_state.current_frame is not None:
                # Display frame with clickable points
                img = st.session_state.current_frame.copy()
                
                # Draw existing points
                if frame_idx in st.session_state.annotations:
                    for point in st.session_state.annotations[frame_idx]:
                        x, y = int(point["x_img"]), int(point["y_img"])
                        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                        cv2.putText(img, point["pitch_id"], (x+10, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                st.image(img, use_container_width=True)
                
                # Click detection (simplified - using coordinates input)
                st.markdown("### Add Point")
                col_x, col_y = st.columns(2)
                with col_x:
                    x_img = st.number_input("X (image)", min_value=0, value=0, key="x_input")
                with col_y:
                    y_img = st.number_input("Y (image)", min_value=0, value=0, key="y_input")
                
                if st.button("Add Point"):
                    if frame_idx not in st.session_state.annotations:
                        st.session_state.annotations[frame_idx] = []
                    
                    st.session_state.annotations[frame_idx].append({
                        "pitch_id": selected_pitch_id,
                        "x_img": float(x_img),
                        "y_img": float(y_img)
                    })
                    st.rerun()
                
                # Show current annotations for this frame
                if frame_idx in st.session_state.annotations:
                    st.markdown(f"**Points for frame {frame_idx}:**")
                    for i, point in enumerate(st.session_state.annotations[frame_idx]):
                        st.text(f"{i+1}. {point['pitch_id']}: ({point['x_img']:.1f}, {point['y_img']:.1f})")
                
                if st.button("Save Annotation for This Frame"):
                    st.success(f"✅ Saved {len(st.session_state.annotations.get(frame_idx, []))} points for frame {frame_idx}")
            else:
                st.info("Click 'Load Frame' to load a frame for annotation")
        
        # Show all annotated frames
        if st.session_state.annotations:
            st.markdown("---")
            st.markdown("### Annotated Frames")
            annotated_frames = sorted(st.session_state.annotations.keys())
            st.write(f"Frames with annotations: {annotated_frames}")

# Step 4: Compute Homographies
elif step == "4. Compute Homographies":
    st.header("Step 4: Compute Homographies")
    
    if not st.session_state.video_id:
        st.warning("⚠️ Please upload a video first (Step 1)")
    elif not st.session_state.annotations:
        st.warning("⚠️ Please annotate pitch points first (Step 3)")
    else:
        st.info(f"Found annotations for {len(st.session_state.annotations)} frames")
        
        if st.button("Compute Homographies"):
            with st.spinner("Computing homographies..."):
                # Convert annotations to API format
                annotations_list = []
                for frame_idx, points in st.session_state.annotations.items():
                    annotations_list.append({
                        "frame_idx": frame_idx,
                        "points": points
                    })
                
                result = call_api(
                    f"/videos/{st.session_state.video_id}/homographies",
                    method="POST",
                    data=annotations_list
                )
                
                if result:
                    st.success(f"✅ Homographies computed for {len(result['frames'])} frames")
                    st.write(f"Frames: {result['frames']}")

# Step 5: Map Players
elif step == "5. Map Players":
    st.header("Step 5: Map Players to Pitch")
    
    if not st.session_state.video_id:
        st.warning("⚠️ Please upload a video first (Step 1)")
    else:
        if st.button("Map Players to Pitch"):
            with st.spinner("Mapping players..."):
                result = call_api(
                    f"/videos/{st.session_state.video_id}/map_players",
                    method="POST"
                )
                
                if result:
                    st.session_state.player_positions = result
                    st.success(f"✅ Mapped {len(result)} player positions")
                    st.info(f"Positions with source='homography': {sum(1 for p in result if p['source'] == 'homography')}")

# Step 6: Interpolate
elif step == "6. Interpolate":
    st.header("Step 6: Interpolate Trajectories")
    
    if not st.session_state.video_id:
        st.warning("⚠️ Please upload a video first (Step 1)")
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_frame = st.number_input("Start Frame", min_value=0, value=0)
        with col2:
            end_frame = st.number_input("End Frame", min_value=0, value=100)
        
        if st.button("Interpolate Trajectories"):
            with st.spinner("Interpolating..."):
                result = call_api(
                    f"/videos/{st.session_state.video_id}/interpolate",
                    method="POST",
                    data={},
                    params={"start_frame": start_frame, "end_frame": end_frame}
                )
                
                if result:
                    st.success(f"✅ Generated {result['frames_generated']} interpolated positions")
                    st.info(f"Method: {result['method']}")

# Step 7: View Results
elif step == "7. View Results":
    st.header("Step 7: View Player Positions")
    
    if not st.session_state.video_id:
        st.warning("⚠️ Please upload a video first (Step 1)")
    else:
        if st.button("Load All Player Positions"):
            with st.spinner("Loading positions..."):
                result = call_api(f"/videos/{st.session_state.video_id}/players", method="GET")
                
                if result:
                    st.session_state.player_positions = result
                    st.success(f"✅ Loaded {len(result)} player positions")
        
        if st.session_state.player_positions:
            # Filter by frame
            frame_filter = st.number_input(
                "Filter by Frame",
                min_value=0,
                value=0,
                help="Enter frame index to filter positions"
            )
            
            filtered = [
                p for p in st.session_state.player_positions
                if p["frame_idx"] == frame_filter
            ]
            
            if filtered:
                st.info(f"Found {len(filtered)} positions for frame {frame_filter}")
                
                # Display as table
                import pandas as pd
                df = pd.DataFrame(filtered)
                st.dataframe(df[["frame_idx", "track_id", "x_pitch", "y_pitch", "source"]])
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Positions", len(filtered))
                with col2:
                    st.metric("Unique Tracks", df["track_id"].nunique())
                with col3:
                    st.metric("Homography", sum(1 for p in filtered if p["source"] == "homography"))
            else:
                st.warning(f"No positions found for frame {frame_filter}")
            
            # Download as JSON
            if st.button("Download Positions (JSON)"):
                json_str = json.dumps(st.session_state.player_positions, indent=2)
                st.download_button(
                    label="Download",
                    data=json_str,
                    file_name=f"positions_{st.session_state.video_id}.json",
                    mime="application/json"
                )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Backend Status")
if st.sidebar.button("Check Backend"):
    try:
        response = requests.get(f"{API_URL}/docs")
        st.sidebar.success("✅ Backend is running")
    except:
        st.sidebar.error("❌ Backend is not running")
