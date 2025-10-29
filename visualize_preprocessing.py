"""
Visualization script for thermal gesture preprocessing steps.
This script visualizes the thermal data transformation through each preprocessing stage
and creates MP4 videos for each step showing a single gesture.

Usage:
    python visualize_preprocessing.py <json_file> [output_dir]
    
Example:
    python visualize_preprocessing.py thermal.json ./preprocessing_visualizations
"""

import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Import preprocessing functions
from src.data.preprocessing import (
    temporal_downsample_to_target,
    subtract_background,
    extract_hand_region,
    calculate_motion_features_raw,
    normalize_motion_frames,
    preprocess_sequence
)


class ThermalPreprocessingVisualizer:
    """Visualize thermal gesture preprocessing steps with MP4 output."""
    
    def __init__(self, output_dir="./preprocessing_visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save MP4 videos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Colormap settings
        self.cmap = plt.cm.inferno
        self.figsize = (14, 10)
        self.fps = 10  # Frames per second for video
        self.dpi = 100
        
    def load_first_gesture(self, json_file):
        """
        Load the first gesture from the JSON file.
        
        Args:
            json_file: Path to thermal data JSON file
            
        Returns:
            List of thermal frames (24x32 arrays) for the first gesture
        """
        print(f"Loading thermal data from {json_file}...")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None
        
        if not data or len(data) == 0:
            print("No data found in JSON file")
            return None
        
        # Extract all frames from all gestures in the JSON file
        all_frames = []
        
        for idx, gesture in enumerate(data):
            if not gesture or len(gesture) < 2:
                print(f"Gesture {idx} is malformed, skipping")
                continue
            
            timestamp = gesture[0]
            raw_data = gesture[1]
            
            if idx == 0:
                print(f"First gesture timestamp: {timestamp}")
                print(f"Raw data length: {len(raw_data) if raw_data else 0}")
            
            # The raw_data contains 768 values (24x32) per timestamp
            # Extract thermal frame from this single frame
            thermal_frames = self._extract_frames_from_raw_data(raw_data)
            all_frames.extend(thermal_frames)
        
        print(f"Extracted {len(all_frames)} total thermal frames from {len(data)} gestures")
        
        return all_frames
    
    def _extract_frames_from_raw_data(self, raw_data):
        """
        Extract multiple thermal frames from raw sensor data.
        
        The raw data may contain multiple 24x32 frame readings.
        This function extracts ALL individual frames without truncation.
        
        Args:
            raw_data: List of raw sensor values
            
        Returns:
            List of thermal frames (24x32 arrays)
        """
        if not raw_data:
            return []
        
        raw_array = np.array(raw_data, dtype=np.float32)
        
        # Standard thermal frame is 24x32 = 768 values
        frame_size = 768
        
        frames = []
        
        # Try to extract frames (assuming data might contain multiple frames)
        # If we have exactly 768 values, treat as single frame
        if len(raw_array) == frame_size:
            frame = raw_array.reshape(24, 32)
            frames.append(frame)
        # If we have more, extract ALL complete frames (no truncation)
        elif len(raw_array) > frame_size:
            # Extract every complete 768-value frame
            num_possible_frames = len(raw_array) // frame_size
            
            # Extract ALL frames, not just first 5!
            for i in range(num_possible_frames):
                start_idx = i * frame_size
                end_idx = start_idx + frame_size
                if end_idx <= len(raw_array):
                    frame = raw_array[start_idx:end_idx].reshape(24, 32)
                    frames.append(frame)
        
        # If we still don't have frames, try alternative extraction
        if not frames and len(raw_array) >= frame_size:
            # Try to find continuous valid thermal data
            # Thermal sensor values are typically in range [0, 65535]
            frame = raw_array[:frame_size].reshape(24, 32)
            frames.append(frame)
        
        return frames
    
    def get_frame_stats(self, frames, label=""):
        """Print statistics about thermal frames."""
        if not frames:
            print(f"{label}: No frames")
            return
        
        all_values = np.concatenate([f.flatten() for f in frames])
        print(f"\n{label} Statistics:")
        print(f"  Frames: {len(frames)}")
        print(f"  Shape per frame: {frames[0].shape}")
        print(f"  Min value: {all_values.min():.2f}")
        print(f"  Max value: {all_values.max():.2f}")
        print(f"  Mean value: {all_values.mean():.2f}")
        print(f"  Median value: {np.median(all_values):.2f}")
    
    def save_video_from_frames(self, frames, output_file, title, cmap=None, is_normalized=False):
        """
        Create MP4 video from thermal frames.
        
        Args:
            frames: List of 2D or 3D thermal frames
            output_file: Path to save MP4 file
            title: Title for the visualization
            cmap: Colormap to use (default: inferno)
            is_normalized: If True, assumes frames are already normalized to 0-1 (preserves segregation)
        """
        if cmap is None:
            cmap = self.cmap
        
        if not frames:
            print(f"No frames to visualize for {title}")
            return
        
        print(f"\nCreating video: {output_file}")
        
        # Handle 3D frames (with channels)
        if frames[0].ndim == 3:
            # If 3 channels, take only first channel
            if frames[0].shape[2] >= 3:
                frames_2d = [f[:, :, 0] for f in frames]
            else:
                frames_2d = [f[:, :, 0] if f.shape[2] > 0 else f[:, :] for f in frames]
        else:
            frames_2d = frames
        
        # Get min/max for scaling
        if is_normalized:
            # For already normalized frames, use fixed range (preserves hand/background segregation)
            vmin = 0
            vmax = 1
        else:
            # For raw frames, find min/max to preserve contrast
            all_values = np.concatenate([f.flatten() for f in frames_2d])
            vmin = np.percentile(all_values, 1)  # Use percentile to ignore outliers
            vmax = np.percentile(all_values, 99)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Initial image
        im = ax.imshow(frames_2d[0], cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, label='Value')
        
        title_text = ax.set_title(f"{title}\nFrame 1/{len(frames_2d)}", fontsize=14, fontweight='bold')
        ax.set_xlabel("X Position (pixels)")
        ax.set_ylabel("Y Position (pixels)")
        
        # Update function for animation
        def update_frame(frame_idx):
            im.set_array(frames_2d[frame_idx])
            title_text.set_text(f"{title}\nFrame {frame_idx + 1}/{len(frames_2d)}")
            im.set_clim(vmin, vmax)
            return [im]
        
        # Create animation
        anim = FuncAnimation(
            fig, update_frame,
            frames=len(frames_2d),
            interval=1000 // self.fps,  # Convert fps to interval in ms
            blit=True,
            repeat=True
        )
        
        # Save as MP4
        try:
            writer = FFMpegWriter(fps=self.fps, metadata=dict(artist='ThermalVisualizer'))
            anim.save(output_file, writer=writer, dpi=self.dpi)
            print(f"✓ Video saved: {output_file}")
        except Exception as e:
            print(f"✗ Error saving video: {e}")
            print("  Make sure ffmpeg is installed and in PATH")
        
        plt.close(fig)
    
    def create_comparison_video(self, frames_dict, output_file, title="Preprocessing Comparison"):
        """
        Create a comparison video showing multiple preprocessing stages side-by-side.
        
        Args:
            frames_dict: Dictionary with stage names as keys and frame lists as values
            output_file: Path to save MP4 file
            title: Title for the visualization
        """
        print(f"\nCreating comparison video: {output_file}")
        
        # Get stage names in order
        stages = list(frames_dict.keys())
        num_stages = len(stages)
        
        if num_stages == 0:
            print("No stages to compare")
            return
        
        # Get number of frames (should be same for all stages)
        num_frames = len(frames_dict[stages[0]])
        
        # Handle 3D frames
        processed_stages = {}
        for stage_name, frames in frames_dict.items():
            if frames[0].ndim == 3:
                if frames[0].shape[2] >= 3:
                    frames_2d = [f[:, :, 0] for f in frames]
                else:
                    frames_2d = [f[:, :, 0] if f.shape[2] > 0 else f[:, :] for f in frames]
            else:
                frames_2d = frames
            processed_stages[stage_name] = frames_2d
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, (num_stages + 1) // 2, figsize=(16, 10), dpi=self.dpi)
        axes = axes.flatten()
        
        # Get global min/max for all stages combined
        all_values = []
        for frames_list in processed_stages.values():
            all_values.extend([f.flatten() for f in frames_list])
        all_values = np.concatenate(all_values)
        vmin = np.percentile(all_values, 1)
        vmax = np.percentile(all_values, 99)
        
        # Create images for each stage
        images = {}
        for idx, (stage_name, frames) in enumerate(processed_stages.items()):
            im = axes[idx].imshow(frames[0], cmap=self.cmap, vmin=vmin, vmax=vmax)
            axes[idx].set_title(stage_name, fontsize=12, fontweight='bold')
            axes[idx].set_xlabel("X")
            axes[idx].set_ylabel("Y")
            plt.colorbar(im, ax=axes[idx], label='Value')
            images[stage_name] = im
        
        # Hide unused subplots
        for idx in range(num_stages, len(axes)):
            axes[idx].set_visible(False)
        
        # Add frame counter
        fig.suptitle(f"{title} - Frame 1/{num_frames}", fontsize=14, fontweight='bold')
        
        # Update function for animation
        def update_frame(frame_idx):
            for stage_name, im in images.items():
                im.set_array(processed_stages[stage_name][frame_idx])
            fig.suptitle(f"{title} - Frame {frame_idx + 1}/{num_frames}", 
                        fontsize=14, fontweight='bold')
            return list(images.values())
        
        # Create animation
        anim = FuncAnimation(
            fig, update_frame,
            frames=num_frames,
            interval=1000 // self.fps,
            blit=True,
            repeat=True
        )
        
        # Save as MP4
        try:
            writer = FFMpegWriter(fps=self.fps, metadata=dict(artist='ThermalVisualizer'))
            anim.save(output_file, writer=writer, dpi=self.dpi)
            print(f"✓ Comparison video saved: {output_file}")
        except Exception as e:
            print(f"✗ Error saving comparison video: {e}")
        
        plt.close(fig)
    
    def visualize_preprocessing_steps(self, thermal_frames, background_percentile=20, hand_percentile_threshold=70):
        """
        Visualize all preprocessing steps and create MP4 videos.
        
        Args:
            thermal_frames: List of raw thermal frames (24x32)
            background_percentile: Percentile for background subtraction (lower = more aggressive)
            hand_percentile_threshold: Percentile for hand region extraction (higher = only hottest regions)
        """
        if not thermal_frames:
            print("No thermal frames to visualize")
            return
        
        print("\n" + "="*70)
        print("THERMAL GESTURE PREPROCESSING VISUALIZATION")
        print("="*70)
        print(f"\n⚙️  Parameters:")
        print(f"  - Background removal percentile: {background_percentile}% (removes coolest {background_percentile}%)")
        print(f"  - Hand extraction threshold: {hand_percentile_threshold}% (keeps only hottest {100-hand_percentile_threshold}%)")
        
        # ===== STEP 0: Original Raw Data =====
        self.get_frame_stats(thermal_frames, "Step 0: Raw Thermal Data")
        output_file = os.path.join(self.output_dir, "00_raw_thermal.mp4")
        self.save_video_from_frames(thermal_frames, output_file, "Step 0: Raw Thermal Data")
        
        # ===== STEP 1: Temporal Resampling =====
        print("\nStep 1: Temporal Resampling to 100 frames...")
        resampled = temporal_downsample_to_target(thermal_frames, target_length=100)
        self.get_frame_stats(resampled, "Step 1: Resampled to 100 frames")
        output_file = os.path.join(self.output_dir, "01_resampled.mp4")
        self.save_video_from_frames(resampled, output_file, "Step 1: Temporal Resampling (100 frames)")
        
        # ===== STEP 2: Background Subtraction =====
        print(f"\nStep 2: Background Subtraction (percentile={background_percentile})...")
        bg_subtracted = subtract_background(resampled, background_percentile=background_percentile)
        self.get_frame_stats(bg_subtracted, "Step 2: Background Subtracted")
        output_file = os.path.join(self.output_dir, "02_background_subtracted.mp4")
        self.save_video_from_frames(bg_subtracted, output_file, "Step 2: Background Subtraction")
        
        # ===== STEP 3: Hand Region Extraction =====
        print(f"\nStep 3: Hand Region Extraction (threshold={hand_percentile_threshold})...")
        hand_regions = [extract_hand_region(frame, percentile_threshold=hand_percentile_threshold) for frame in bg_subtracted]
        self.get_frame_stats(hand_regions, "Step 3: Hand Region Extracted")
        output_file = os.path.join(self.output_dir, "03_hand_region.mp4")
        self.save_video_from_frames(hand_regions, output_file, "Step 3: Hand Region Extraction")
        
        # ===== STEP 4: Motion Features Calculation =====
        print("\nStep 4: Motion Features Calculation (Thermal Only)...")
        motion_frames = calculate_motion_features_raw(hand_regions)
        print(f"Motion frames shape: {motion_frames[0].shape}")
        print(f"  Channel 0 (Thermal): {motion_frames[0][:, :, 0].min():.2f} to {motion_frames[0][:, :, 0].max():.2f}")
        
        # Visualize thermal channel only
        thermal_channel = [f[:, :, 0] for f in motion_frames]
        
        self.get_frame_stats(thermal_channel, "Step 4: Thermal Channel")
        output_file = os.path.join(self.output_dir, "04_thermal_channel.mp4")
        self.save_video_from_frames(thermal_channel, output_file, "Step 4: Motion Features - Thermal Channel")
        
        # ===== STEP 5: Normalization =====
        print("\nStep 5: Motion Frame Normalization...")
        normalized_frames = normalize_motion_frames(motion_frames)
        print(f"Normalized frames shape: {normalized_frames[0].shape}")
        print(f"  Channel 0 (Thermal) normalized: {normalized_frames[0][:, :, 0].min():.2f} to {normalized_frames[0][:, :, 0].max():.2f}")
        
        # Visualize normalized thermal channel
        thermal_norm = [f[:, :, 0] for f in normalized_frames]
        
        self.get_frame_stats(thermal_norm, "Step 5: Normalized Thermal Channel")
        output_file = os.path.join(self.output_dir, "05_thermal_normalized.mp4")
        self.save_video_from_frames(thermal_norm, output_file, "Step 5: Normalized Thermal Channel [0-1]", 
                                   is_normalized=True)
        
        print("\n" + "="*70)
        print("✓ VISUALIZATION COMPLETE")
        print(f"✓ All videos saved to: {os.path.abspath(self.output_dir)}")
        print("="*70 + "\n")


def main():
    """Main function to run the visualization."""
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python visualize_preprocessing.py <json_file> [output_dir]")
        print("\nExample:")
        print("  python visualize_preprocessing.py thermal.json ./preprocessing_visualizations")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./preprocessing_visualizations"
    
    # Validate input file
    if not os.path.exists(json_file):
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    
    # Create visualizer and run
    visualizer = ThermalPreprocessingVisualizer(output_dir=output_dir)
    
    # Load first gesture
    thermal_frames = visualizer.load_first_gesture(json_file)
    
    if not thermal_frames:
        print("Failed to load thermal frames")
        sys.exit(1)
    
    # Visualize preprocessing steps
    visualizer.visualize_preprocessing_steps(thermal_frames)


if __name__ == "__main__":
    main()
