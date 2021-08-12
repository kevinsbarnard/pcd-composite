import glob
import os
import signal

import open3d as o3d
import numpy as np
import fire

DOWNSAMPLE_VOXEL_SIZE = 2


def find_raytrix_pcd_files(pcd_dir: str):
    pcd_files = glob.glob(os.path.join(pcd_dir, '*.pcd'))
    pcd_files.sort(key=lambda v: int(os.path.basename(v).split('_')[3]))  # Sort by frame number
    return pcd_files


def rolling_composite_registration(pcds, threshold, voxel_size=DOWNSAMPLE_VOXEL_SIZE, visualize=False):
    transforms = [np.eye(4)]
    composite_pcd = pcds[0]
    composite_pcd.estimate_normals()
    
    # Initial guess at registration transform (uncomment one)
    # transform_guess = lambda: np.eye(4)  # Identity transform
    transform_guess = lambda: transforms[-1]  # Last transform
    
    if visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(composite_pcd)
    
    for idx, pcd in enumerate(pcds[1:]):
        registration = o3d.pipelines.registration.registration_icp(
            pcd, composite_pcd,
            threshold,
            transform_guess(),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10)
        )
        transform = registration.transformation
        transforms.append(transform)
        
        if visualize:
            vis.remove_geometry(composite_pcd)
        
        # Transform point cloud and append to composite
        pcd.transform(transform)
        composite_pcd += pcd
        
        # Downsample the composite and re-estimate normals
        # composite_pcd, _ = composite_pcd.remove_radius_outlier(5, 20)
        # composite_pcd, _ = composite_pcd.remove_statistical_outlier(20, 2.0)
        composite_pcd = composite_pcd.voxel_down_sample(voxel_size=3*voxel_size)
        composite_pcd.estimate_normals()
        
        if visualize:
            vis.add_geometry(composite_pcd)
            vis.poll_events()
            vis.update_renderer()
    
    if visualize:
        while vis.poll_events():
            vis.update_renderer()
            
        vis.destroy_window()
    
    return transforms


def main(pcd_dir: str):
    os.setpgrp()
    try:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        pcd_files = find_raytrix_pcd_files(pcd_dir)
        pcds = [o3d.io.read_point_cloud(pcd_file) for pcd_file in pcd_files[::5]]
        print(len(pcds))
        transforms = rolling_composite_registration(pcds, 1e4, visualize=True)
    finally:
        os.killpg(0, signal.SIGKILL)


if __name__ == '__main__':
    fire.Fire(main)
