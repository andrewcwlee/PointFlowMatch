#!/usr/bin/env python3
"""
Replay demo data and visualize point clouds with attention bounding boxes.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torch

from pfp import DATA_DIRS
from pfp.data.replay_buffer import RobotReplayBuffer
from pfp.data.dataset_pcd_attention import RobotDatasetPcdAttention, points_in_bbox
from pfp.common.attention_visualization import (
    visualize_attention_matplotlib, 
    visualize_attention_o3d,
    create_attention_video,
    plot_attention_statistics
)

# Import exact bbox extraction functions
try:
    from scripts.exact_bbox_extraction_for_replay import (
        extract_exact_bboxes_from_replay_data,
        extract_exact_object_points_from_masks
    )
    HAS_EXACT_EXTRACTION = True
except ImportError:
    print("Warning: exact_bbox_extraction_for_replay not found - using approximate method")
    HAS_EXACT_EXTRACTION = False

# Add this for raw point cloud processing
try:
    import open3d as o3d
    from pfp.common.o3d_utils import make_pcd
    HAS_O3D = True
except ImportError:
    HAS_O3D = False
    print("Warning: Open3D not available, cannot load raw point clouds")


def extract_object_bbox_from_masks(pcd, masks, object_id, camera_names):
    """
    Extract bounding box for an object using segmentation masks.
    Returns bbox center and size, or None if object not found.
    """
    # Count pixels for this object across all cameras
    total_pixels = 0
    for i, camera in enumerate(camera_names):
        if i < len(masks):
            mask = masks[i]
            obj_pixels = (mask == object_id).sum()
            total_pixels += obj_pixels
    
    if total_pixels == 0:
        return None
    
    # Use spatial heuristics to estimate object location
    # For objects of interest [31, 34, 35, 92], use broader height ranges
    if object_id in [31, 34, 35]:  # Small connector/gripper parts
        height_mask = (pcd[:, 2] >= 0.7) & (pcd[:, 2] <= 1.3)
        cluster_radius = 0.15
    elif object_id == 92:  # Another small object
        height_mask = (pcd[:, 2] >= 0.7) & (pcd[:, 2] <= 1.3)
        cluster_radius = 0.15
    else:
        height_mask = (pcd[:, 2] >= 0.6) & (pcd[:, 2] <= 1.4)
        cluster_radius = 0.3
    
    candidate_points = pcd[height_mask]
    
    if len(candidate_points) == 0:
        return None
    
    # Cluster points around centroid
    centroid = candidate_points.mean(axis=0)
    distances = np.linalg.norm(candidate_points - centroid, axis=1)
    cluster_mask = distances < cluster_radius
    final_points = candidate_points[cluster_mask]
    
    if len(final_points) < 3:
        final_points = candidate_points  # Use all candidate points
        
    if len(final_points) < 3:
        return None
    
    # Compute bounding box
    min_bound = final_points.min(axis=0)
    max_bound = final_points.max(axis=0)
    center = (min_bound + max_bound) / 2
    size = max_bound - min_bound
    
    return {
        'center': center,
        'size': size,
        'min': min_bound,
        'max': max_bound,
        'n_points': len(final_points)
    }



def visualize_object_bboxes_from_masks(pcd: np.ndarray, masks: np.ndarray, 
                                      object_ids: list = [31, 34, 35, 92]):
    """
    Visualize bounding boxes for specific objects from segmentation masks.
    """
    camera_names = ['right_shoulder', 'left_shoulder', 'overhead', 'front', 'wrist']
    
    # Extract bounding boxes for objects of interest
    object_bboxes = {}
    for obj_id in object_ids:
        bbox = extract_object_bbox_from_masks(pcd, masks, obj_id, camera_names)
        if bbox is not None:
            object_bboxes[obj_id] = bbox
    
    return object_bboxes


def visualize_gripper_bbox(pcd: np.ndarray, gripper_pos: np.ndarray, padding: float = 0.1):
    """Visualize 3D gripper bounding box on point cloud."""
    # Create 3D bounding box around gripper
    bbox_min = gripper_pos - padding
    bbox_max = gripper_pos + padding
    
    # Check which points are inside the 3D bounding box
    inside_x = (pcd[:, 0] >= bbox_min[0]) & (pcd[:, 0] <= bbox_max[0])
    inside_y = (pcd[:, 1] >= bbox_min[1]) & (pcd[:, 1] <= bbox_max[1])
    inside_z = (pcd[:, 2] >= bbox_min[2]) & (pcd[:, 2] <= bbox_max[2])
    
    attention_weights = (inside_x & inside_y & inside_z).astype(np.float32)
    return attention_weights


def create_attention_mask_from_object_points(pcd: np.ndarray, object_points: dict, 
                                            gripper_pos: np.ndarray = None, 
                                            distance_threshold: float = 0.01):
    """
    Create attention mask where points belonging to objects (or near them) have weight 1.
    
    Args:
        pcd: Point cloud (N, 3)
        object_points: Dict of {object_id: exact 3D points array}
        gripper_pos: Optional gripper position to include
        distance_threshold: Distance threshold for considering a point as belonging to an object
        
    Returns:
        attention_mask: (N,) array with 1 for object points, 0 otherwise
    """
    attention_mask = np.zeros(len(pcd), dtype=np.float32)
    
    # Combine all object points into one array for efficient distance computation
    all_object_points = []
    for obj_id, obj_pts in object_points.items():
        if len(obj_pts) > 0:
            all_object_points.append(obj_pts)
    
    if len(all_object_points) > 0:
        all_object_points = np.vstack(all_object_points)
        
        # Vectorized distance computation using broadcasting
        # For each point in pcd, find minimum distance to any object point
        # This is more efficient than nested loops
        for i in range(0, len(pcd), 100):  # Process in batches to manage memory
            batch_end = min(i + 100, len(pcd))
            batch_pcd = pcd[i:batch_end]
            
            # Compute distances from batch points to all object points
            # Shape: (batch_size, num_object_points)
            distances = np.linalg.norm(
                batch_pcd[:, None, :] - all_object_points[None, :, :], 
                axis=2
            )
            
            # Find minimum distance for each point in batch
            min_distances = distances.min(axis=1)
            
            # Set attention to 1 for points close to objects
            mask = min_distances < distance_threshold
            attention_mask[i:batch_end][mask] = 1.0
    
    # Include gripper region if provided
    if gripper_pos is not None:
        # Add small region around gripper
        gripper_distances = np.linalg.norm(pcd - gripper_pos[None, :], axis=1)
        gripper_mask = gripper_distances < 0.05  # 5cm radius around gripper
        attention_mask[gripper_mask] = 1.0
    
    return attention_mask


def replay_demo(
    data_path: str,
    episode_idx: int = 0,
    output_dir: str = "./demo_visualization",
    bbox_padding: float = 0.1,
    vis_mode: str = "matplotlib",  # "matplotlib", "o3d", "video"
    n_points: int = 4096,
    use_raw_pcd: bool = False,  # New parameter to use raw uncropped point clouds
):
    """
    Replay a demo episode and visualize point clouds with attention.
    
    Args:
        data_path: Path to demo data directory
        episode_idx: Which episode to replay (default: 0)
        output_dir: Directory to save visualizations
        bbox_padding: Padding for gripper bounding box (only used when no masks available)
        vis_mode: Visualization mode
        n_points: Number of points to sample
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Loading demo data from: {data_path}")
    
    # Load replay buffer
    replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="r")
    
    # Get episode data
    episode_indices = replay_buffer.episode_ends
    if episode_idx >= len(episode_indices):
        print(f"Episode {episode_idx} not found. Available episodes: 0-{len(episode_indices)-1}")
        return
    
    # Get episode start and end
    start_idx = 0 if episode_idx == 0 else episode_indices[episode_idx - 1]
    end_idx = episode_indices[episode_idx]
    episode_length = end_idx - start_idx
    
    print(f"Episode {episode_idx}: {episode_length} steps")
    
    # Load episode data
    pcd_data = replay_buffer["pcd_xyz"][start_idx:end_idx]  # (T, N, 3)
    robot_states = replay_buffer["robot_state"][start_idx:end_idx]  # (T, 10)
    
    if "pcd_color" in replay_buffer.keys():
        pcd_colors = replay_buffer["pcd_color"][start_idx:end_idx]  # (T, N, 3)
        pcd_colors = pcd_colors.astype(np.float32) / 255.0
    else:
        pcd_colors = None
    
    # Check if segmentation masks and camera point clouds are available
    if "segmentation_masks" in replay_buffer.keys():
        segmentation_masks = replay_buffer["segmentation_masks"][start_idx:end_idx]  # (T, 5, H, W)
        print(f"Segmentation masks shape: {segmentation_masks.shape}")
        has_masks = True
        object_ids_of_interest = [31, 34, 35, 92]
        
        # Check for individual camera point clouds (for exact bbox extraction)
        if "camera_point_clouds" in replay_buffer.keys():
            camera_point_clouds = replay_buffer["camera_point_clouds"][start_idx:end_idx]  # (T, 5, H*W, 3)
            print(f"Camera point clouds shape: {camera_point_clouds.shape}")
            has_exact_extraction = True
        else:
            camera_point_clouds = None
            has_exact_extraction = False
            print("No individual camera point clouds - using approximate spatial filtering")
    else:
        segmentation_masks = None
        camera_point_clouds = None
        has_masks = False
        has_exact_extraction = False
        print("No segmentation masks found - using gripper-based attention only")
    
    print(f"Point cloud shape: {pcd_data.shape}")
    print(f"Robot state shape: {robot_states.shape}")
    
    # Sample points if needed
    if pcd_data.shape[1] > n_points:
        print(f"Sampling {n_points} points from {pcd_data.shape[1]}")
        random_indices = np.random.choice(pcd_data.shape[1], n_points, replace=False)
        pcd_data = pcd_data[:, random_indices]
        if pcd_colors is not None:
            pcd_colors = pcd_colors[:, random_indices]
    
    # Generate attention masks for each timestep based on combined bounding box
    attention_masks = []
    gripper_positions = []
    combined_bboxes = []  # Store single combined bbox for all timesteps
    
    for t in range(episode_length):
        gripper_pos = robot_states[t, :3]  # Extract gripper position
        gripper_positions.append(gripper_pos)
        
        # Create single combined bounding box covering gripper + objects of interest
        if has_masks and t < len(segmentation_masks):
            if has_exact_extraction and HAS_EXACT_EXTRACTION and t < len(camera_point_clouds):
                # Use exact extraction with individual camera point clouds
                try:
                    # Extract exact object points from masks
                    object_points = extract_exact_object_points_from_masks(
                        camera_point_clouds[t], segmentation_masks[t], 
                        object_ids_of_interest, debug=(t == 0)
                    )
                    
                    # Create attention mask directly from object points
                    attention_mask = create_attention_mask_from_object_points(
                        pcd_data[t], object_points, gripper_pos, distance_threshold=0.01
                    )
                    
                    combined_bboxes.append(None)  # No bbox needed
                    
                    if t == 0:  # Debug info for first frame
                        print(f"  Frame {t} - Direct mask extraction:")
                        total_object_points = sum(len(pts) for pts in object_points.values())
                        print(f"    Objects found: {list(object_points.keys())}")
                        print(f"    Total object points: {total_object_points}")
                        print(f"    Attention points: {attention_mask.sum()}/{len(attention_mask)}")
                        for obj_id, pts in object_points.items():
                            if len(pts) > 0:
                                print(f"    Object {obj_id}: {len(pts)} exact points")
                        
                except Exception as e:
                    print(f"  Frame {t} - Exact extraction failed: {e}, falling back to gripper-only")
                    # Fall back to gripper-only attention
                    attention_mask = visualize_gripper_bbox(pcd_data[t], gripper_pos, bbox_padding)
                    combined_bboxes.append(None)
            else:
                # No individual camera point clouds - fall back to gripper-only
                if t == 0:
                    print(f"  Frame {t} - No camera point clouds available, using gripper-only attention")
                attention_mask = visualize_gripper_bbox(pcd_data[t], gripper_pos, bbox_padding)
                combined_bboxes.append(None)
        else:
            # Fallback to gripper-based attention if no masks
            attention_mask = visualize_gripper_bbox(pcd_data[t], gripper_pos, bbox_padding)
            combined_bboxes.append(None)
        
        attention_masks.append(attention_mask)
    
    attention_masks = np.array(attention_masks)  # (T, N)
    gripper_positions = np.array(gripper_positions)  # (T, 3)
    
    if has_masks:
        if has_exact_extraction and HAS_EXACT_EXTRACTION:
            print(f"Generated attention masks from exact object segmentation with {attention_masks.mean():.2%} average attention")
        else:
            print(f"Generated attention masks from approximate object detection with {attention_masks.mean():.2%} average attention")
    else:
        print(f"Generated gripper-based attention masks with {attention_masks.mean():.2%} average attention")
    
    # Visualization
    if vis_mode == "matplotlib":
        # Create individual frame visualizations
        for t in tqdm(range(min(10, episode_length)), desc="Creating visualizations"):
            colors = pcd_colors[t] if pcd_colors is not None else None
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Original point cloud
            ax1 = fig.add_subplot(121, projection='3d')
            if colors is not None:
                ax1.scatter(pcd_data[t, :, 0], pcd_data[t, :, 1], pcd_data[t, :, 2], 
                           c=colors, s=1, alpha=0.6)
            else:
                ax1.scatter(pcd_data[t, :, 0], pcd_data[t, :, 1], pcd_data[t, :, 2], 
                           s=1, alpha=0.6)
            
            # Add gripper position
            ax1.scatter(*gripper_positions[t], c='red', s=100, marker='x', linewidths=3)
            ax1.set_title(f"Original Point Cloud (t={t})")
            ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
            
            # Set fixed axis limits
            x_range = [-0.3, 0.4]
            y_range = [-0.2, 0.3]
            z_range = [0.6, 1.5]
            ax1.set_xlim(x_range); ax1.set_ylim(y_range); ax1.set_zlim(z_range)
            ax1.set_box_aspect([1,1,1])  # Equal aspect ratio
            
            # Attention visualization
            ax2 = fig.add_subplot(122, projection='3d')
            colors_att = plt.cm.viridis(attention_masks[t])[:, :3]
            ax2.scatter(pcd_data[t, :, 0], pcd_data[t, :, 1], pcd_data[t, :, 2], 
                       c=colors_att, s=2, alpha=0.8)
            
            # Add gripper position
            ax2.scatter(*gripper_positions[t], c='red', s=100, marker='*', linewidths=3)
            
            # No bounding box drawing needed - attention is shown by point colors
            
            ax2.set_title(f"Attention Weights (t={t})")
            ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
            
            # Set same axis limits for consistency
            ax2.set_xlim(x_range); ax2.set_ylim(y_range); ax2.set_zlim(z_range)
            ax2.set_box_aspect([1,1,1])  # Equal aspect ratio
            
            # Set camera view angle (elevation, azimuth)
            ax1.view_init(elev=20, azim=45)  # Default view
            ax2.view_init(elev=20, azim=45)  # Same view for consistency
            
            plt.tight_layout()
            plt.savefig(output_path / f"demo_attention_t{t:03d}.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Saved visualizations to {output_path}")
    
    elif vis_mode == "o3d":
        # Interactive Open3D visualization
        try:
            import open3d as o3d
            for t in range(min(5, episode_length)):
                print(f"Showing timestep {t} (close window to continue)")
                visualize_attention_o3d(
                    pcd_data[t],
                    attention_masks[t],
                    pcd_colors[t] if pcd_colors is not None else None,
                    threshold=0.5
                )
        except ImportError:
            print("Open3D not available, falling back to matplotlib")
            vis_mode = "matplotlib"
    
    elif vis_mode == "video":
        # Create video with dual view (original + attention)
        video_path = output_path / "demo_attention_video.mp4"
        print(f"Creating video: {video_path}")
        
        # Create dual-view video frames
        import tempfile
        import imageio.v2 as imageio
        import os
        
        max_frames = episode_length  # Limit for reasonable video size
        temp_dir = Path(tempfile.mkdtemp())
        frame_paths = []
        
        try:
            for t in tqdm(range(max_frames), desc="Generating video frames"):
                fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                
                # Original point cloud
                ax1 = fig.add_subplot(121, projection='3d')
                if pcd_colors is not None:
                    ax1.scatter(pcd_data[t, :, 0], pcd_data[t, :, 1], pcd_data[t, :, 2], 
                               c=pcd_colors[t], s=3, alpha=0.7)
                else:
                    ax1.scatter(pcd_data[t, :, 0], pcd_data[t, :, 1], pcd_data[t, :, 2], 
                               c='blue', s=3, alpha=0.7)
                
                # Add gripper position
                ax1.scatter(*gripper_positions[t], c='red', s=200, marker='x', linewidths=4)
                ax1.set_title(f"Original Point Cloud (Step {t+1}/{max_frames})", fontsize=14)
                ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
                
                # Set fixed axis limits
                x_range = [-0.3, 0.4]
                y_range = [-0.2, 0.3]
                z_range = [0.6, 1.5]
                ax1.set_xlim(x_range); ax1.set_ylim(y_range); ax1.set_zlim(z_range)
                ax1.set_box_aspect([1,1,1])  # Equal aspect ratio
                
                # Attention visualization
                ax2 = fig.add_subplot(122, projection='3d')
                colors_att = plt.cm.viridis(attention_masks[t])[:, :3]
                ax2.scatter(pcd_data[t, :, 0], pcd_data[t, :, 1], pcd_data[t, :, 2], 
                           c=colors_att, s=5, alpha=0.8)
                
                # Add gripper position
                ax2.scatter(*gripper_positions[t], c='red', s=200, marker='*', linewidths=4)
                
                # No bounding box drawing needed - attention is shown by point colors
                
                ax2.set_title(f"Attention Weights (Step {t+1}/{max_frames})", fontsize=14)
                ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
                
                # Set same axis limits for consistency
                ax2.set_xlim(x_range); ax2.set_ylim(y_range); ax2.set_zlim(z_range)
                ax2.set_box_aspect([1,1,1])  # Equal aspect ratio
                
                # Add colorbar for attention
                mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
                mappable.set_array(attention_masks[t])
                cbar = plt.colorbar(mappable, ax=ax2, shrink=0.5, aspect=10)
                cbar.set_label('Attention Weight', rotation=270, labelpad=15)
                
                plt.tight_layout()
                
                # Save frame to both temp and output directory
                frame_path = temp_dir / f"frame_{t:04d}.png"
                output_frame_path = output_path / f"frame_{t:04d}.png"
                plt.savefig(frame_path, dpi=100, bbox_inches='tight')
                plt.savefig(output_frame_path, dpi=100, bbox_inches='tight')
                plt.close()
                frame_paths.append(frame_path)
            
            print(f"Frames saved. Creating video from {len(frame_paths)} frames...")
            
            # Create video from saved frames in output directory
            frame_files = sorted(list(output_path.glob("frame_*.png")))
            print(f"Found {len(frame_files)} frame files")
            
            if len(frame_files) > 0:
                # Try ffmpeg first if available
                try:
                    import subprocess
                    ffmpeg_path = output_path / "demo_attention_video_ffmpeg.mp4"
                    cmd = [
                        'ffmpeg', '-y', '-r', '8', 
                        '-pattern_type', 'glob', '-i', str(output_path / 'frame_*.png'),
                        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', str(ffmpeg_path)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"Video saved with ffmpeg: {ffmpeg_path}")
                    else:
                        raise Exception("ffmpeg failed")
                except:
                    # Fallback to imageio
                    print("Trying imageio for video creation...")
                    try:
                        images = []
                        for frame_file in frame_files:
                            image = imageio.imread(str(frame_file))
                            # Ensure image is RGB and valid
                            if image is not None and len(image.shape) == 3:
                                if image.shape[-1] == 4:
                                    image = image[:, :, :3]
                                # Convert to uint8 if needed
                                if image.dtype != np.uint8:
                                    image = (image * 255).astype(np.uint8)
                                images.append(image)
                        
                        if len(images) > 0:
                            # Create GIF (more reliable than MP4)
                            gif_path = output_path / "demo_attention_video.gif"
                            imageio.mimsave(str(gif_path), images, fps=6, duration=1/6)
                            print(f"GIF saved: {gif_path}")
                        else:
                            print("No valid images found for video creation")
                            
                    except Exception as e2:
                        print(f"Video creation failed: {e2}")
                        print("Individual frame files are available in:", output_path)
            else:
                print("No frame files found. Check output directory:", output_path)
            
        finally:
            # Clean up temporary files
            for frame_path in frame_paths:
                if frame_path.exists():
                    frame_path.unlink()
            temp_dir.rmdir()
    
    # Plot attention statistics
    stats_path = output_path / "attention_statistics.png"
    plot_attention_statistics(attention_masks, save_path=str(stats_path))
    
    # Save summary info
    summary_path = output_path / "demo_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Demo Episode {episode_idx} Summary\n")
        f.write(f"="*40 + "\n")
        f.write(f"Episode length: {episode_length} steps\n")
        f.write(f"Point cloud shape: {pcd_data.shape}\n")
        f.write(f"Average attention: {attention_masks.mean():.2%}\n")
        f.write(f"Attention std: {attention_masks.std():.4f}\n")
        f.write(f"Points in attention region: {(attention_masks > 0).sum()} / {attention_masks.size}\n")
        f.write(f"\nGripper trajectory:\n")
        f.write(f"Start position: {gripper_positions[0]}\n")
        f.write(f"End position: {gripper_positions[-1]}\n")
        f.write(f"Total distance: {np.linalg.norm(gripper_positions[-1] - gripper_positions[0]):.3f}m\n")
    
    print(f"\nDemo replay complete!")
    print(f"Visualizations saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Replay demo with attention visualization")
    parser.add_argument("--data_path", type=str, 
                       default="demos/sim/unplug_charger/train_segmented",
                       help="Path to demo data directory (use train_segmented for object bboxes)")
    parser.add_argument("--episode", type=int, default=0,
                       help="Episode index to replay")
    parser.add_argument("--output_dir", type=str, 
                       default="./demo_visualization",
                       help="Output directory for visualizations")
    parser.add_argument("--bbox_padding", type=float, default=0.1,
                       help="Gripper bounding box padding (meters)")
    parser.add_argument("--vis_mode", type=str, default="video",
                       choices=["matplotlib", "o3d", "video"],
                       help="Visualization mode")
    parser.add_argument("--n_points", type=int, default=4096,
                       help="Number of points to sample")
    
    args = parser.parse_args()
    
    # Convert relative path to absolute
    if not Path(args.data_path).is_absolute():
        args.data_path = Path.cwd() / args.data_path
    
    if not Path(args.data_path).exists():
        print(f"Data path does not exist: {args.data_path}")
        return
    
    replay_demo(
        data_path=str(args.data_path),
        episode_idx=args.episode,
        output_dir=args.output_dir,
        bbox_padding=args.bbox_padding,
        vis_mode=args.vis_mode,
        n_points=args.n_points,
    )


if __name__ == "__main__":
    main()