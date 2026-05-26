"""
1. Opens two TIFF images:
* clipped_pre.tif
* clipped_post.tif
2. Checks that they match spatially
3. Splits both images into smaller 518x518 pixel tiles
4. Saves each tile as a separate GeoTIFF
5. Keeps georeferencing information for every tile
"""

# Imports libraries
import os 
import rasterio
from rasterio.windows import Window

# Input files
pre_tif = "clipped_pre.tif"
post_tif = "clipped_post.tif"

# Output folders
pre_output = "pre_tiles"
post_output = "post_tiles"

# Create folders if they don't exist
os.makedirs(pre_output, exist_ok=True)
os.makedirs(post_output, exist_ok=True)

# Tile size
tile_size = 518

# Open both raster files 
with rasterio.open(pre_tif) as pre_src, rasterio.open(post_tif) as post_src:
   
    # Check compability, ensures that both images align perfectly 
    assert pre_src.width == post_src.width, "Width mismatch"
    assert pre_src.height == post_src.height, "Height mismatch"
    assert pre_src.transform == post_src.transform, "Transform mismatch"
    assert pre_src.crs == post_src.crs, "CRS mismatch"

    # Get image dimensions
    width = pre_src.width
    height = pre_src.height

    # Print image size
    print(f"Imagesize: {width} x {height}")

    # Tile counter
    tile_id = 0

    # Loop through image
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):

            # Handle edge tiles, prevents reading outside the image, ex last tile might be 212x518
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)

            # Create raster window
            window = Window(x, y, w, h)

            # Read PRE tile
            pre_tile = pre_src.read(window=window)

            # Compute tile transform
            pre_transform = pre_src.window_transform(window)

            # Copy metadata profile
            pre_profile = pre_src.profile.copy()
            
            # Update metadata
            pre_profile.update({
                "height": h,
                "width": w,
                "transform": pre_transform,
            })

            # Create output filename
            pre_output_path = os.path.join(
                pre_output,
                f"tile_{tile_id:04d}.tif"
            )
            
            # Write PRE tile
            with rasterio.open(pre_output_path, "w", **pre_profile) as dst:
                dst.write(pre_tile)
            
            # Repeat for POST image
            post_tile = post_src.read(window=window)

            post_transform = post_src.window_transform(window)

            post_profile = post_src.profile.copy()
            post_profile.update({
                "height": h, 
                "width": w,
                "transform": post_transform
            })

            post_output_path = os.path.join(
                post_output,
                f"tile_{tile_id:04d}.tif"
            )

            with rasterio.open(post_output_path, "w", **post_profile) as dst:
                dst.write(post_tile)

            # Print progress
            print(f"Saved tile {tile_id:04d}")

            # Increas the tile ID
            tile_id += 1
print("Finished!")



