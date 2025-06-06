"""Visualization utilities for attention weights on point clouds."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from typing import Optional, Union


def visualize_attention_matplotlib(
    pcd: np.ndarray,
    attention_weights: np.ndarray,
    title: str = "Point Cloud with Attention Weights",
    figsize: tuple = (10, 8),
    view_angle: tuple = (20, 45),
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize point cloud with attention weights using matplotlib.
    
    Args:
        pcd: Point cloud coordinates (N, 3)
        attention_weights: Attention weights for each point (N,)
        title: Plot title
        figsize: Figure size
        view_angle: (elevation, azimuth) viewing angles
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize attention weights to [0, 1]
    weights_norm = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-8)
    
    # Create colormap
    colors = plt.cm.hot(weights_norm)
    
    # Plot points
    scatter = ax.scatter(
        pcd[:, 0], pcd[:, 1], pcd[:, 2],
        c=colors,
        s=20,
        alpha=0.8,
        edgecolors='none'
    )
    
    # Add colorbar
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.hot)
    mappable.set_array(attention_weights)
    cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set viewing angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_attention_o3d(
    pcd: np.ndarray,
    attention_weights: np.ndarray,
    pcd_colors: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    show_axes: bool = True,
) -> None:
    """
    Visualize point cloud with attention weights using Open3D.
    
    Args:
        pcd: Point cloud coordinates (N, 3)
        attention_weights: Attention weights for each point (N,)
        pcd_colors: Optional original point colors (N, 3)
        threshold: Attention threshold for highlighting
        show_axes: Whether to show coordinate axes
    """
    # Create Open3D point cloud
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
    
    # Normalize attention weights
    weights_norm = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-8)
    
    # Create colors based on attention
    if pcd_colors is not None:
        # Blend original colors with attention
        attention_color = np.array([1.0, 0.0, 0.0])  # Red for high attention
        colors = pcd_colors * (1 - weights_norm[:, None]) + attention_color * weights_norm[:, None]
    else:
        # Use colormap
        colors = plt.cm.hot(weights_norm)[:, :3]  # Remove alpha channel
    
    o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create visualization list
    vis_list = [o3d_pcd]
    
    # Add coordinate axes
    if show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis_list.append(axes)
    
    # Optionally add bounding box around high attention regions
    high_attention_mask = weights_norm > threshold
    if high_attention_mask.any():
        high_attention_points = pcd[high_attention_mask]
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(high_attention_points)
        )
        bbox.color = (0, 1, 0)  # Green bounding box
        vis_list.append(bbox)
    
    # Visualize
    o3d.visualization.draw_geometries(
        vis_list,
        window_name="Point Cloud Attention Visualization",
        width=1024,
        height=768,
    )


def create_attention_video(
    pcds: np.ndarray,
    attention_weights: np.ndarray,
    output_path: str,
    fps: int = 10,
    view_angle: tuple = (20, 45),
) -> None:
    """
    Create a video of point clouds with attention weights over time.
    
    Args:
        pcds: Point clouds over time (T, N, 3)
        attention_weights: Attention weights over time (T, N)
        output_path: Path to save the video
        fps: Frames per second
        view_angle: (elevation, azimuth) viewing angles
    """
    import imageio
    import tempfile
    import os
    
    T = pcds.shape[0]
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Generate frames
        frame_paths = []
        for t in range(T):
            frame_path = os.path.join(temp_dir, f"frame_{t:04d}.png")
            visualize_attention_matplotlib(
                pcds[t],
                attention_weights[t],
                title=f"Time Step {t}",
                save_path=frame_path,
                view_angle=view_angle,
            )
            frame_paths.append(frame_path)
        
        # Create video
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)
                
        print(f"Video saved to: {output_path}")
        
    finally:
        # Clean up temporary files
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                os.remove(frame_path)
        os.rmdir(temp_dir)


def plot_attention_statistics(
    attention_weights: Union[np.ndarray, torch.Tensor],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot statistics of attention weights.
    
    Args:
        attention_weights: Attention weights (*, N) where * can be any batch dimensions
        save_path: Optional path to save the figure
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Flatten all batch dimensions
    weights_flat = attention_weights.flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram
    axes[0, 0].hist(weights_flat, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Attention Weight')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Distribution of Attention Weights')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_weights = np.sort(weights_flat)
    cumsum = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)
    axes[0, 1].plot(sorted_weights, cumsum, linewidth=2)
    axes[0, 1].set_xlabel('Attention Weight')
    axes[0, 1].set_ylabel('Cumulative Probability')
    axes[0, 1].set_title('Cumulative Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot
    if len(attention_weights.shape) > 1:
        # Show box plot for each time step or batch
        reshaped = attention_weights.reshape(-1, attention_weights.shape[-1])
        axes[1, 0].boxplot(reshaped.T, showfliers=False)
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Attention Weight')
        axes[1, 0].set_title('Attention Distribution per Sample')
    else:
        axes[1, 0].boxplot(weights_flat, showfliers=False)
        axes[1, 0].set_ylabel('Attention Weight')
        axes[1, 0].set_title('Overall Attention Distribution')
    
    # Statistics text
    stats_text = f"Mean: {weights_flat.mean():.4f}\n"
    stats_text += f"Std: {weights_flat.std():.4f}\n"
    stats_text += f"Min: {weights_flat.min():.4f}\n"
    stats_text += f"Max: {weights_flat.max():.4f}\n"
    stats_text += f"Median: {np.median(weights_flat):.4f}\n"
    
    # Calculate sparsity (percentage of weights < 0.1)
    sparsity = (weights_flat < 0.1).sum() / len(weights_flat)
    stats_text += f"Sparsity (<0.1): {sparsity:.2%}"
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    transform=axes[1, 1].transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Statistics')
    
    plt.suptitle('Attention Weight Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    N = 1000  # Number of points
    
    # Generate example point cloud
    pcd = np.random.randn(N, 3)
    
    # Generate example attention weights (higher for points near origin)
    distances = np.linalg.norm(pcd, axis=1)
    attention_weights = np.exp(-distances)
    attention_weights = attention_weights / attention_weights.max()
    
    # Visualize with matplotlib
    visualize_attention_matplotlib(pcd, attention_weights, title="Example Attention Visualization")
    
    # Plot statistics
    plot_attention_statistics(attention_weights)