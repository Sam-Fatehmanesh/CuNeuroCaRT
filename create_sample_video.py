import tifffile
import numpy as np
import cv2
import os
from pathlib import Path
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_video(tiff_file, output_file, num_frames=600, fps=20, z_slice=15):
    """
    Create a sample video from a TIFF file.
    
    Parameters:
    -----------
    tiff_file : str
        Path to the TIFF file
    output_file : str
        Path to the output video file
    num_frames : int
        Number of frames to include in the video
    fps : int
        Frames per second for the output video
    z_slice : int
        Z-slice to use for the video (middle slice by default)
    """
    logger.info(f"Reading TIFF file: {tiff_file}")
    
    # Memory map the TIFF file to avoid loading the entire file into memory
    with tifffile.TiffFile(tiff_file) as tif:
        series = tif.series[0]
        logger.info(f"TIFF series shape: {series.shape}")
        
        # Determine the number of z-slices per volume
        # From the filename, we know there are 30 slices
        z_slices_per_volume = 30
        
        # Calculate the number of time points
        time_points = series.shape[0] // z_slices_per_volume
        logger.info(f"Z-slices per volume: {z_slices_per_volume}")
        logger.info(f"Total time points: {time_points}")
        
        # Limit the number of frames to process
        frames_to_process = min(num_frames, time_points)
        logger.info(f"Processing {frames_to_process} frames")
        
        # Get the shape of a single frame
        frame_height, frame_width = series.shape[1], series.shape[2]
        
        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        
        logger.info(f"Creating video: {output_file}")
        
        # Create a memory-mapped array without loading all data at once
        data = tifffile.memmap(tiff_file)
        
        # Process each time point with a progress bar
        for t in tqdm(range(frames_to_process), desc="Creating video", unit="frame"):
            # Calculate the index for the desired z-slice at this time point
            idx = t * z_slices_per_volume + z_slice
            
            # Read the frame (only loads this specific frame into memory)
            frame = data[idx]
            
            # Normalize to 8-bit for video
            frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_normalized, cv2.COLOR_GRAY2BGR)
            
            # Write to video
            video.write(frame_bgr)
        
        # Release the video writer
        video.release()
        
        logger.info(f"Video created successfully: {output_file}")

if __name__ == "__main__":
    # Input TIFF file
    tiff_file = "20230331_HuC_GRAB_5HT4_fish1_5uWside_50msExp_30slices_4x4bin_fullRun_30minE3_30minDMSO_1hr_MT_1.tif"
    
    # Output video file
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "sample_video_600frames_20fps.avi")
    
    # Create the sample video
    create_sample_video(tiff_file, output_file, num_frames=600, fps=20) 