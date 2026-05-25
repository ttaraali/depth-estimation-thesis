"""
The following script was developed in Python to automate depth estimation and
change detection for slected satellite image tiles using the Depth Anything V2 framework.
The implementation relies on PyTorch, Rasterio, NumPy, and the Depth Anything V2 model 
architecture.

Created 19 May 2026
"""

# Import required libraries
import os 
import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Import Depth Anything V2 model
from depth_anything_v2.dpt import DepthAnythingV2

# Select computation device (GPU, Apple Silicon, or CPU)
DEVICE = ('cuda' if torch.cuda.is_available()
	else 'mps' if torch.backends.mps.is_available()
	else 'cpu')

# Define model configurations for different encoder sizes
model_configs = {
	'vits': {'encoder': 'vits',
		  'features': 64,
		  'out_channels': [48, 96, 192, 384]
	},
	'vitb': {'encoder': 'vitb',
		'features': 128,
		'out_channels': [96, 192, 384, 768]
	},
    'vitl': {'encoder': 'vitl',
		'features': 256,
		'out_channels': [256, 512, 1024, 1024]
	},
    'vitg': {'encoder': 'vitg',
		'features': 384,
		'out_channels': [1536, 1536, 1536, 1536]
	}
}

# Select encoder architecture 
encoder = 'vitl' 

# Initialize and load pretrained model weights 
model = DepthAnythingV2(**model_configs[encoder])

model.load_state_dict(
	torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', 
			map_location='cpu'
	)
)

# Move model to selected device and set evaluation mode
model = model.to(DEVICE).eval()

# Define input and output directories
pre_folder = "pre_tiles"
post_folder = "post_tiles"

pre_depth_folder = "pre_depth_maps"
post_depth_folder = "post_depth_maps"
diff_folder = "depth_difference"

# Create output directories if they do not exist
os.makedirs(pre_depth_folder, exist_ok=True)
os.makedirs(post_depth_folder, exist_ok=True)
os.makedirs(diff_folder, exist_ok=True)

# Selected TIFF tiles used for analysis
selected_tiles = [
	"tile_0165.tif",
	"tile_0186.tif",
	"tile_0371.tif",
	"tile_0372.tif",
	"tile_0369.tif",
	"tile_0435.tif"
]

# Process each selected tile
for tile_name in selected_tiles:

    pre_path = os.path.join(pre_folder, tile_name)
    post_path = os.path.join(post_folder, tile_name)

    # Verify that corresponding pre-event and post-event tiles exist
    if not os.path.exists(pre_path):
            print(f"Missing PRE tile for {tile_name}")
            continue

    if not os.path.exists(post_path):
            print(f"Missing POST tile for {tile_name}")
            continue
    
    # Read pre-event raster image
    with rasterio.open(pre_path) as src:
        pre_img = src.read()
        profile = src.profile 

    # Convert raster format from (bands, rows, columns) to (rows, columns, bands)
    pre_img = pre_img.transpose(1,2,0)

    # Retain RGB bands only 
    if pre_img.shape[2] >= 3: 
        pre_img = pre_img[:, :, :3]
    
    # Normalize pixel values to 8-bit range
    pre_img = pre_img.astype(np.float32)

    pre_img = (
        (pre_img - pre_img.min()) / 
        (pre_img.max() - pre_img.min())
        ) * 255

    pre_img = pre_img.astype(np.uint8)

    # Read post-event raster image
    with rasterio.open(post_path) as src:
        post_img = src.read()
        profile = src.profile 

    post_img = post_img.transpose(1,2,0)

    if post_img.shape[2] >= 3: 
        post_img = post_img[:, :, :3]

    post_img = post_img.astype(np.float32)

    post_img = (
        (post_img - post_img.min()) / 
        (post_img.max() - post_img.min())
        ) * 255

    post_img = post_img.astype(np.uint8)

	# Generate monocular depth maps 
    pre_depth = model.infer_image(pre_img)
    post_depth = model.infer_image(post_img)
    
    # Compute depth difference between post-event and pre-event imagery
    depth_diff = post_depth - pre_depth

    # Define common visulization range for depth maps
    combined_min = min(pre_depth.min(), post_depth.min())
    combined_max = max(pre_depth.max(), post_depth.max())

    # Update raster metadata profile for output depth maps
    profile.update(
		dtype=rasterio.float32,
		count=1
	)
	
    # Export pre-event depth raster
    with rasterio.open(
        os.path.join(
            pre_depth_folder, 
            f"pre_depth_{tile_name}"
        ),
        "w",
        **profile
    ) as dst:

        dst.write(pre_depth.astype(rasterio.float32), 1)
    
    # Export post-event depth raster
    with rasterio.open(
	    os.path.join(
	      post_depth_folder, 
	      f"post_depth_{tile_name}"
	    ),
	    "w",
	    **profile
    ) as dst:

        dst.write(post_depth.astype(rasterio.float32), 1)
    
    # Create visualization figure
    fig, axes = plt.subplots(
	    3,
	    2, 
	    figsize=(14,18)
	)
    
    # Display RGB imagery
    axes[0, 0].imshow(pre_img)
    axes[0, 0].set_title("PRE RGB")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(post_img)
    axes[0, 1].set_title("POST RGB")
    axes[0, 1].axis("off")

    # Display pre-event depth map
    im1 = axes[1, 0].imshow(
	      pre_depth,
	      cmap="inferno",
	      vmin=combined_min,
	      vmax=combined_max
	)
    
    axes[1, 0].set_title("PRE Depth Map")
    axes[1, 0]. axis("off")

    
    # Add colorbar for pre-event depth visualization
    cbar1 = fig.colorbar(
	       im1, 
	       ax=axes[1, 0], 
	       fraction=0.046,
	       pad=0.04
	)
	
    cbar1.set_label("Depth")


    # Display post-event depth map
    im2 = axes[1, 1].imshow(
	      post_depth,
	      cmap="inferno",
	      vmin=combined_min,
	      vmax=combined_max
	)
	
    axes[1, 1].set_title("POST Depth Map")
    axes[1, 1]. axis("off")

    # Add colorbar for post-event depth visualization
    cbar2 = fig.colorbar(
	       im2, 
	       ax=axes[1, 1], 
	       fraction=0.046,
	       pad=0.04
	)
	
    cbar2.set_label("Depth")


    # Visualize depth difference between post-event and pre-event imagery
    im3 = axes[2, 0].imshow(
	     depth_diff,
	     cmap="bwr",
	     vmin= depth_diff.min(), 
	     vmax=depth_diff.max()
	)
	
    axes[2, 0].set_title("Depth Difference (POST-PRE)")
    axes[2, 0].axis("off")
	
    # Add colorbar for depth difference visualization
    cbar3 = fig.colorbar(
	      im3,
	      ax=axes[2, 0],
	      fraction=0.046,
	      pad=0.04
	)
	
    cbar3.set_label("Depth Difference")


    # Disable unused subplot panel
    axes[2, 1].axis("off")

    # Add overall figure tile     
    plt.suptitle(
	     f"Tile Analysis:{tile_name}",
	     fontsize=18
	)
	
    # Adjust subplot layout
    plt.tight_layout()
	
    # Define output path for visualization figure
    combined_output_path = os.path.join(
		diff_folder,
		f"combined_visualization_{tile_name}.png"
	)
	
    # Export visulization figure
    plt.savefig(combined_output_path,
		     dpi=300,
		     bbox_inches="tight"
    )

    # Close figure to reduce memory usage
    plt.close()

    # Print processing status
    print(f"Processed and visualized: {tile_name}")	
