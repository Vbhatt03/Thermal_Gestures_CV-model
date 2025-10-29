#!/usr/bin/env python
"""
Quick-start script for visualizing thermal preprocessing.
Run this script to visualize the preprocessing pipeline with a single command.
"""

import os
import sys

# Refresh PATH to include newly installed FFmpeg
if sys.platform == "win32":
    path_machine = os.environ.get("Path", "")
    path_user = os.environ.get("Path", "")
    try:
        import ctypes
        env_var = ctypes.windll.kernel32.GetEnvironmentVariableW
        size = env_var("Path", None, 0)
        buf = ctypes.create_unicode_buffer(size)
        env_var("Path", buf, size)
        os.environ["Path"] = buf.value
    except:
        pass

from visualize_preprocessing import ThermalPreprocessingVisualizer


def run_visualization():
    """Run visualization with default settings."""
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                 THERMAL GESTURE PREPROCESSING VISUALIZER                 ║
║                                                                           ║
║  This will create MP4 videos showing each preprocessing step applied      ║
║  to the first gesture in your thermal data.                              ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Find thermal.json in current directory
    json_file = "A_1.json"
    
    if not os.path.exists(json_file):
        print(f"❌ Error: {json_file} not found in current directory")
        print(f"   Current directory: {os.getcwd()}")
        print(f"\nPlease run this script from the project root directory where thermal.json is located.")
        return False
    
    output_dir = "preprocessing_visualizations"
    
    print(f"📂 Input file: {os.path.abspath(json_file)}")
    print(f"📁 Output directory: {os.path.abspath(output_dir)}")
    print()
    
    try:
        visualizer = ThermalPreprocessingVisualizer(output_dir=output_dir)
        
        # Load first gesture
        print("Loading thermal data...")
        thermal_frames = visualizer.load_first_gesture(json_file)
        
        if not thermal_frames:
            print("❌ Failed to load thermal frames")
            return False
        
        print(f"✓ Loaded {len(thermal_frames)} frames")
        
        # Run visualization with improved hand isolation parameters
        print("\nGenerating videos (this may take 5-10 minutes)...\n")
        print("📋 Parameters for better hand isolation:")
        print("   - background_percentile: 10 (removes coolest 10%, keeps hottest 90%)")
        print("   - hand_percentile_threshold: 85 (keeps only hottest 15% as hand region)")
        print()
        
        visualizer.visualize_preprocessing_steps(
            thermal_frames,
            background_percentile=5,      # More aggressive background removal
            hand_percentile_threshold=85   # Stricter hand region (only hottest pixels)
        )
        
        print("✓ DONE!")
        print(f"\nVideos are ready to view in: {os.path.abspath(output_dir)}")
        print("\nNext steps:")
        print(f"  1. Open the output folder: {output_dir}")
        print(f"  2. Start with: 00_raw_thermal.mp4")
        print(f"  3. Watch in order to see each preprocessing step")
        print(f"  4. View: 06_all_stages_comparison.mp4 for side-by-side comparison")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_visualization()
    sys.exit(0 if success else 1)
