import argparse
from pathlib import Path

import os
import cv2
import numpy as np
from PIL import Image
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent  
sys.path.append(str(ROOT / "Depth-Anything-V2"))

import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from depth_anything_v2.dpt import DepthAnythingV2
import open3d as o3d


# --------------------------------------------------
# 1. Mask2Former 相關：載入 + 推論 + 取最大連通區塊
# --------------------------------------------------
def load_m2f(model_dir, device):
    model_dir = Path(model_dir)
    print(f"[Mask2Former] Loading from: {model_dir}")
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir).to(device)
    model.eval()
    return image_processor, model


def keep_largest_component(mask_uint8):
    """沿用你原本的邏輯：只保留最大的連通區塊（0/255 mask）。"""
    bin_mask = (mask_uint8 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    if num_labels <= 1:
        return mask_uint8

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_id = 1 + np.argmax(areas)

    out = np.zeros_like(mask_uint8, dtype=np.uint8)
    out[labels == largest_id] = 255
    return out


def infer_roi_mask(image_pil, image_processor, model, device):
    """
    給一張 PIL.Image，回傳 bool mask (H, W)，True 表示屬於花床 ROI。
    """
    inputs = image_processor(image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred_semantic_map = image_processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[image_pil.size[::-1]]  # (H, W)
        )[0]  # (H, W), 每個 pixel 是一個 class id

    fg_mask = (pred_semantic_map.cpu().numpy() == 1).astype(np.uint8) * 255
    fg_mask = keep_largest_component(fg_mask)

    roi_bool = fg_mask > 0
    return roi_bool, fg_mask


# --------------------------------------------------
# 2. Depth Anything V2：載入 + 單張推論
# --------------------------------------------------
def load_depth_anything(encoder, device):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }

    print(f"[DepthAnything] Loading encoder: {encoder}")
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    ckpt_path = f'checkpoints/depth_anything_v2_{encoder}.pth'  
    depth_anything.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    depth_anything = depth_anything.to(device).eval()
    return depth_anything


def infer_depth_raw(image_bgr, depth_anything, input_size, device):
    depth_raw = depth_anything.infer_image(image_bgr, input_size)  # float32, (H, W)
    depth_raw = depth_raw.astype(np.float32)
    return depth_raw


# --------------------------------------------------
# 3. point cloud：把 depth + RGB (+ ROI) 轉成 PLY
# --------------------------------------------------
def depth_to_pointcloud_orthographic(depth_map, image_rgb, roi_mask=None, scale=3.0):
    """
    depth_map: float32, (H, W)
    image_rgb: uint8, (H, W, 3), RGB
    roi_mask:  bool,  (H, W)，True 表示要保留的像素
    scale:     depth 縮放係數
    """
    h, w = depth_map.shape

    # 建 pixel grid：x 對應 w，y 對應 h
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # depth 平移 & 縮放（只要相對高低，不管絕對單位）
    d = depth_map - depth_map.min()
    z = d * scale

    points = np.stack([xx, yy, z], axis=-1).reshape(-1, 3).astype(np.float32)

    base_mask = z.reshape(-1) > 0

    if roi_mask is not None:
        roi_flat = roi_mask.reshape(-1)
        base_mask &= roi_flat

    points = points[base_mask]

    colors = image_rgb.reshape(-1, 3)[base_mask] / 255.0

    # 建立 Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 去雜點
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.0)
    inlier_cloud = pcd.select_by_index(ind)

    return inlier_cloud

def estimate_surface_area(pcd, real_long_side_cm=230.0,
                          voxel_ratio=1/200,   # 以物體尺度自動抓 voxel
                          normal_radius_mult=3.0,
                          poisson_depth=12,
                          density_q=0.05,
                          orient_k=50):
    if len(pcd.points) < 500:
        return 0.0, None

    # 0) 離群點移除（先做）
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

    # 1) 自動估 voxel_size（用 AABB 尺度）
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(aabb.get_extent())
    voxel_size = max(extent * voxel_ratio, 1e-6)

    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 2) normals
    normal_radius = voxel_size * normal_radius_mult
    pcd_ds.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=60)
    )
    # 法向一致化（Poisson 穩定度關鍵）
    pcd_ds.orient_normals_consistent_tangent_plane(k=orient_k)

    # 3) Poisson
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_ds, depth=poisson_depth, linear_fit=True
    )

    # 4) density 過濾（用 vertex mask 正確刪）
    densities = np.asarray(densities)
    thr = np.quantile(densities, density_q)
    mesh.remove_vertices_by_mask(densities < thr)

    # 5) mesh 清理
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # 6) 尺度校正（仍用 OBB，但建議你之後換成 marker/ROI）
    obb = mesh.get_oriented_bounding_box()
    virtual_max_len = float(np.max(obb.extent))
    if virtual_max_len <= 0:
        return 0.0, mesh

    scale_factor = real_long_side_cm / virtual_max_len
    mesh.scale(scale_factor, center=mesh.get_center())

    area = float(mesh.get_surface_area())  # 單位會跟你 scale 後一致（這裡是 cm^2）
    return area, mesh


# --------------------------------------------------
# 4. 主流程：自動跑整個 pipeline
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Auto pipeline: ROI seg + depth + 3D reconstruction")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Mask2Former model directory")
    parser.add_argument("--img-dir", type=str, required=True,
                        help="Directory of input images (jpg/png)")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory (will create subfolders: masks, depth, ply)")
    parser.add_argument("--encoder", type=str, default="vits",
                        choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=518,
                        help="DepthAnythingV2 input size")
    parser.add_argument("--scale", type=float, default=3.0,
                        help="Scale for depth to Z in point cloud")
    parser.add_argument("--real-len", type=float, default=230.0,
                    help="Real world max length of the object in cm")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"[Device] Using: {device}")

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    masks_dir = out_dir / "masks"
    depth_dir = out_dir / "depth_raw"
    ply_dir   = out_dir / "ply"

    masks_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)

    # 1) 載入模型
    image_processor, m2f_model = load_m2f(args.model_dir, device)
    depth_anything = load_depth_anything(args.encoder, device)

    # 2) 找出所有影像
    img_paths = sorted([
        p for p in img_dir.glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])
    print(f"[Info] Found {len(img_paths)} images.")

    # 3) 逐張處理
    for i, img_path in enumerate(img_paths, 1):
        print(f"\n[{i}/{len(img_paths)}] Processing: {img_path.name}")
        pil_img = Image.open(img_path).convert("RGB")
        rgb = np.array(pil_img)  # (H, W, 3) RGB

        # 3-1) ROI segmentation
        roi_bool, roi_uint8 = infer_roi_mask(pil_img, image_processor, m2f_model, device)
        mask_save_path = masks_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(mask_save_path), roi_uint8)
        print(f"  - ROI mask saved to: {mask_save_path}")

        # 3-2) Depth estimation
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_raw = infer_depth_raw(bgr, depth_anything, args.input_size, device)
        depth_npy_path = depth_dir / f"{img_path.stem}_raw.npy"
        np.save(depth_npy_path, depth_raw)
        print(f"  - Depth raw saved to: {depth_npy_path}")

        # 3-3) 3D reconstruction → point cloud
        pcd = depth_to_pointcloud_orthographic(
            depth_map=depth_raw,
            image_rgb=rgb,
            roi_mask=roi_bool,
            scale=args.scale
        )

        ply_path = ply_dir / f"{img_path.stem}.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"  - Point cloud saved to: {ply_path}")

        area, mesh = estimate_surface_area(pcd, args.real_len)
        print(f"  - Estimated Surface Area: {area:.2f} cm²")

        if mesh is not None:
            mesh_dir = out_dir / "mesh"
            mesh_dir.mkdir(exist_ok=True)
            mesh_path = mesh_dir / f"{img_path.stem}_mesh.ply"
            o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    
if __name__ == "__main__":
    main()