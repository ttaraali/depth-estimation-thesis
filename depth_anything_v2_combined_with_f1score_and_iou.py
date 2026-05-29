import os 

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from cv2 import resize, INTER_LINEAR    
from sklearn.metrics import f1_score

from depth_anything_v2.dpt import DepthAnythingV2



DEVICE = ('cuda' if torch.cuda.is_available()
	else 'mps' if torch.backends.mps.is_available()
	else 'cpu')

print(f"Using device: {DEVICE}")


model_configs = {

	'vits': {
        'encoder': 'vits',
		  'features': 64,
		  'out_channels': [48, 96, 192, 384]
	},

	'vitb': {
        'encoder': 'vitb',
		'features': 128,
		'out_channels': [96, 192, 384, 768]
	},

    'vitl': {
        'encoder': 'vitl',
		'features': 256,
		'out_channels': [256, 512, 1024, 1024]
	},

    'vitg': {'encoder': 'vitg',
		'features': 384,
		'out_channels': [1536, 1536, 1536, 1536]
	}
}


encoder = 'vitl' 

model = DepthAnythingV2(**model_configs[encoder])

model.load_state_dict(
	torch.load(
        f'checkpoints/depth_anything_v2_{encoder}.pth', 
			map_location='cpu'
	)
)

model = model.to(DEVICE).eval()


pre_folder = "pre_tiles"
post_folder = "post_tiles"


pre_depth_folder = "pre_depth_maps"
post_depth_folder = "post_depth_maps"

diff_folder = "depth_difference"
colored_diff_folder = "colored_depth_difference"


os.makedirs(pre_depth_folder, exist_ok=True)
os.makedirs(post_depth_folder, exist_ok=True)
os.makedirs(diff_folder, exist_ok=True)
os.makedirs(colored_diff_folder, exist_ok=True)


selected_tiles = [
    "tile_0165.tif", 
    "tile_0186.tif",
    "tile_0369.tif",
    "tile_0371.tif", 
    "tile_0372.tif", 
    "tile_0435.tif"
]

all_tile_ground_truth = {
	"tile_0165.tif": [
        (317387.6, 4162203.5),
        (317349.8, 4162107.2),
        (317387.9, 4162104.3),
    ],
	
    "tile_0186.tif": [
        (315875.2, 4161989.9), 
        (315905.6, 4161970.9),
        (315931.4, 4161951.9),
        (315957.8, 4161935.1)
    ],
	
    "tile_0369.tif": [
        (316409.6, 4161130.4), 
        (316390.8, 4161121.7),
        (316448.1, 4161069.3),
        (316469.8, 4161063.0),
        (316455.7, 4161014.2),
        (316448.1, 4160987.6),
        (316413.9, 4161026.7),
        (316410.9, 4160986.9),
        (316382.5, 4161023.9),
        (316383.8, 4160988.8)
    ],

    "tile_0371.tif": [
        (316745.5, 4161130.4),
        (316673.5, 4161108.1),
        (316654.1, 4161083.2),
        (316645.4, 4161044.2),
    ],

	"tile_0372.tif": [
        (316916.3, 4161119.0),
        (316893.3, 4161102.8),
        (316910.8, 4161078.6),
        (316952.9, 4161040.8),
        (316950.5, 4161024.3),
        (316947.1, 4160983.7)
    ],


	"tile_0435.tif": [
        (317384.35, 4160749.44),
        (317391.3, 4160732.4),
        (317396.1, 4160719.8),
        (317404.0, 4160701.2),
        (317408.6, 4160688.2),
        (317416.4, 4160664.6)
    ]
}

thesis_performance_table = {}


for tile_name in selected_tiles:
    print(f"\nProcessing:{tile_name}")

    pre_path = os.path.join(pre_folder, tile_name)
    post_path = os.path.join(post_folder, tile_name)

    if not os.path.exists(pre_path) or not os.path.exists(post_path):
        print(f"Missing image assets for tile: {tile_name}")
        continue
   

    with rasterio.open(pre_path) as src:
        pre_img = src.read()
        profile = src.profile.copy()        
        original_height = src.height
        original_width = src.width


    pre_img = pre_img.transpose(1,2,0).astype(np.float32)

    if pre_img.shape[2] >= 3: 
        pre_img = pre_img[:, :, :3]
    
    pre_img = (
        (pre_img - pre_img.min()) / 
        (pre_img.max() - pre_img.min())
    ) * 255
    
    pre_img = pre_img.astype(np.uint8)

    with rasterio.open(post_path) as src:
        post_img = src.read()

    post_img = post_img.transpose(1,2,0).astype(np.float32)

    if post_img.shape[2] >= 3: 
        post_img = post_img[:, :, :3]

    post_img = (
        (post_img - post_img.min()) / 
        (post_img.max() - post_img.min()) 
        ) * 255
    
    post_img = post_img.astype(np.uint8)


    pre_depth = model.infer_image(pre_img)
    post_depth = model.infer_image(post_img)
    

    pre_depth = resize(
        pre_depth,
        (original_width, original_height),
        interpolation=INTER_LINEAR
    )

    post_depth = resize(
        post_depth,
        (original_width, original_height),
        interpolation=INTER_LINEAR
    )


    depth_diff = post_depth - pre_depth

    pixel_points = all_tile_ground_truth.get(tile_name, [])

    if len(pixel_points) > 0:
        height, width = depth_diff.shape

        ground_truth_mask = np.zeros_like(depth_diff)
        evaluation_mask = np.zeros_like(depth_diff)

        buffer_radius = 22

        with rasterio.open(pre_path) as src:
            for geo_x, geo_y in pixel_points:
                row, col = src.index(geo_x, geo_y)
                
                if 0 <= row < height and 0 <= col < width:
                    ground_truth_mask[
                        max(0, row - buffer_radius): min(
                            row + buffer_radius + 1, 
                            height,
                        ), 
                        max(0, col-buffer_radius): min(
                            col + buffer_radius + 1, 
                            width,
                        ), 
                    ] = 1
            
                    evaluation_mask[
                        max(0, row - buffer_radius): min(
                            row + buffer_radius + 1, 
                            height,
                        ), 
                        max(0, col - buffer_radius): min(
                            col + buffer_radius + 1, 
                            width), 
                    ] = 1



        local_mean = np.mean(depth_diff[evaluation_mask ==1])
        local_std = np.std(depth_diff[evaluation_mask ==1])

        ai_detected_damage = np.where(
            (
                depth_diff <= (local_mean - 0.25 * local_std)
            ) 
            & (evaluation_mask==1), 
            1, 
            0
        )

        points_detected_count = 0
        total_points = len(pixel_points)

        with rasterio.open(pre_path) as src:
            for geo_x, geo_y in pixel_points:
                row, col = src.index(geo_x, geo_y)

                row_idx = min(max(0, row), height - 1)
                col_idx = min(max(0, col), width - 1)

                local_neighborhood = ai_detected_damage[
                    max(0, row_idx - 1): min(row_idx + 2, height),
                    max(0, col_idx - 1): min(col_idx + 2, width),
                ]

                if np.any(local_neighborhood == 1):
                    points_detected_count += 1
                        
        y_true = ground_truth_mask[evaluation_mask==1].flatten()
        y_pred = ai_detected_damage[evaluation_mask==1].flatten()

        intersection = np.logical_and(
            ai_detected_damage, 
            ground_truth_mask
        ).sum()
        
        union = np.logical_or(
            ai_detected_damage, 
            ground_truth_mask
        ).sum()

        iou_score = (
            intersection / union 
            if union > 0 else 0.0
        )

        f1 = f1_score(
            y_true,
            y_pred, 
            zero_division=0
        )

        
        thesis_performance_table[tile_name] = {
            "IoU": iou_score, 
            "F1": f1,
            "Points": f"{points_detected_count}/{total_points}",
        }
        
        print(
            f"Validated {tile_name} ->"
            f"Detected: {points_detected_count}/{total_points} points"
        )

    combined_min = min(
        pre_depth.min(),
        post_depth.min()
    )

    combined_max = max(
        pre_depth.max(),
        post_depth.max()
    )

    profile.update(
        driver="GTiff",
        dtype=rasterio.float32, 
        count=1,
        compress="lzw",
        nodata=np.nan
    )

    with rasterio.open(
        os.path.join(
            pre_depth_folder,
            f"pre_depth_{tile_name}",
        ),
        "w",
        **profile,
    ) as dst:
        dst.write(pre_depth.astype(rasterio.float32),1)

    with rasterio.open(
        os.path.join(
            post_depth_folder,
            f"post_depth_{tile_name}",
        ),
        "w",
        **profile,
    ) as dst:
        dst.write(post_depth.astype(rasterio.float32),1)

  
    with rasterio.open(
        os.path.join(
            diff_folder,
            f"depth_difference_{tile_name}",
        ),
        "w",
        **profile
    ) as dst:
        dst.write(
            depth_diff.astype(rasterio.float32),
            1
        )

    norm_diff = (
        depth_diff - depth_diff.min()
    ) / (
        depth_diff.max() - depth_diff.min()
    )

    colored_diff = cm.bwr(norm_diff)

    colored_diff = (
        colored_diff[:, :, :3] * 255
    ).astype(np.uint8)

    rgb_profile = profile.copy()

    rgb_profile.update(
        driver="GTiff",
        dtype=rasterio.uint8,
        count=3,
        compress="lzw",
        nodata=None
    )



    with rasterio.open(
        os.path.join(
            colored_diff_folder,
            f"colored_depth_difference_{tile_name}"
        ),
        "w",
        **rgb_profile
    ) as dst: 
        
        dst.write(colored_diff[:, :, 0], 1)
        dst.write(colored_diff[:, :, 1], 2)
        dst.write(colored_diff[:, :, 2], 3)


    fig, axes = plt.subplots(
        3,
        2, 
        figsize=(14, 18)
    )

    axes[0,0].imshow(pre_img)
    axes[0,0].set_title("PRE RGB")
    axes[0,0].axis("off")

    axes[0,1].imshow(post_img)
    axes[0,1].set_title("POST RGB")
    axes[0,1].axis("off")

    im1 = axes[1,0].imshow(
        pre_depth,
        cmap="inferno",
        vmin=combined_min,
        vmax=combined_max
    )

    axes[1,0].set_title("PRE Depth Map")
    axes[1,0].axis("off")

    fig.colorbar(
        im1,
        ax=axes[1,0],
        fraction=0.046,
        pad=0.04
    )

    im2 = axes[1,1].imshow(
        post_depth,
        cmap="inferno",
        vmin=combined_min,
        vmax=combined_max
    )

    axes[1,1].set_title("POST Depth Map")
    axes[1,1].axis("off")

    fig.colorbar(
        im2,
        ax=axes[1,1],
        fraction=0.046,
        pad=0.04
    )

    im3 = axes[2,0].imshow(
        depth_diff,
        cmap="bwr",
        vmin=depth_diff.min(),
        vmax=depth_diff.max()
    )

    axes[2,0].set_title("Depth Difference (POST - PRE)")
    axes[2,0].axis("off")

    fig.colorbar(
        im3,
        ax=axes[2,0],
        fraction=0.046,
        pad=0.04
    )   
    axes[2,1].axis("off")

    plt.tight_layout()

    combined_output_path = os.path.join(
        diff_folder,
        f"combined_visualization_{tile_name}.png"
    )

    plt.savefig(
        combined_output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Finished processing: {tile_name}")



print("\n" + "="*80)
print("Final validation summary")
print("="*80)

print(
    f"{'Tile Identifier':<20}"
    f"{'IoU Score':<18}"
    f"{'F1-Score':<18}"
    f"{'Points Detected':<15}"

)


print("-"*80)

for t_name, metrics in thesis_performance_table.items():
    
    print(
        f"{t_name:<20} " 
        f"{metrics['IoU']:<18.4f}"
        f"{metrics['F1']:<18.4f}"
        f"{metrics['Points']:<15}"
    )
    
print("="*70)
