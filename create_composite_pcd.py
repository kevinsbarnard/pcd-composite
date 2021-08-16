import glob
import os
from typing import List

import open3d as o3d
import progressbar
import fire


def find_pcd_files(input_directory: str) -> List[str]:
    return glob.glob(os.path.join(input_directory, '*.pcd'))


def build_composite(pcd_files: List[str]) -> o3d.geometry.PointCloud:
    composite_pcd = o3d.geometry.PointCloud()
    print('Building composite pointcloud...')
    for pcd_file in progressbar.progressbar(pcd_files):
        composite_pcd += o3d.io.read_point_cloud(pcd_file)
    
    return composite_pcd


def create_composite(input_directory: str, output: str = 'composite.pcd'):
    pcd_files = find_pcd_files(input_directory)
    composite_pcd = build_composite(pcd_files)
    o3d.io.write_point_cloud(output, composite_pcd)
    

if __name__ == '__main__':
    fire.Fire(create_composite)