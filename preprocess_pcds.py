import glob
import os
from typing import Optional, List

import progressbar
import open3d as o3d
import numpy as np
import fire


def find_raytrix_pcd_files(pcd_in_dir):
    pcd_files = glob.glob(os.path.join(pcd_in_dir, '*.pcd'))
    pcd_files.sort(key=lambda v: int(os.path.basename(v).split('_')[3]))  # Sort by frame number
    return pcd_files


class PointCloudOperation:
    def run(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        raise NotImplementedError
    
    def __str__(self):
        raise NotImplementedError
    

class ScaleOperation(PointCloudOperation):
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        
    @property
    def tf(self):
        tf = np.eye(4, dtype=np.float64)
        tf[0, 0] = self.x
        tf[1, 1] = self.y
        tf[2, 2] = self.z
        
        return tf
    
    def run(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pcd.transform(self.tf)
        return pcd
    
    def __str__(self):
        return 'Scale: x={} y={} z={}'.format(self.x, self.y, self.z)


class VoxelDownsampleOperation(PointCloudOperation):
    def __init__(self, voxel_size: float):
        self.voxel_size = voxel_size
    
    def run(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        return pcd.voxel_down_sample(voxel_size=self.voxel_size)
    
    def __str__(self):
        return 'Voxel downsample: size {}'.format(self.voxel_size)
    

class UniformDownsampleOperation(PointCloudOperation):
    def __init__(self, every_k_points: int):
        self.every_k_points = every_k_points
    
    def run(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        return pcd.uniform_down_sample(every_k_points=self.every_k_points)
    
    def __str__(self):
        return 'Uniform downsample: every {} points'.format(self.every_k_points)
    

class StatisticalOutlierRemovalOperation(PointCloudOperation):
    def __init__(self, nb_neighbors: int, std_ratio: float):
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
    
    def run(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pcd_out, _ = pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio)
        return pcd_out
    
    def __str__(self):
        return 'Statistical outlier removal: consider {} neighbors with standard deviation ratio {}'.format(self.nb_neighbors, self.std_ratio)
    
    
class RadiusOutlierRemovalOperation(PointCloudOperation):
    def __init__(self, nb_points: int, radius: float):
        self.nb_points = nb_points
        self.radius = radius
    
    def run(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pcd_out, _ = pcd.remove_radius_outlier(nb_points=self.nb_points, raidus=self.radius)
        return pcd_out
    
    def __str__(self):
        return 'Radius outlier removal: minimum {} points within a radius of {}'.format(self.nb_points, self.radius)


class BatchPointCloudProcessor:
    def __init__(self, in_dir: str, out_dir: str):
        self.pcd_files = find_raytrix_pcd_files(in_dir)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.steps = []
    
    def scale(self, x: float, y: float, z: float):
        self.steps.append(ScaleOperation(x, y, z))
        return self
    
    def voxel_downsample(self, voxel_size: float):
        self.steps.append(VoxelDownsampleOperation(voxel_size))
        return self
    
    def uniform_downsample(self, every_k_points: float):
        self.steps.append(UniformDownsampleOperation(every_k_points))
        return self
    
    def statistical_outlier_removal(self, nb_neighbors: int, std_ratio: float):
        self.steps.append(StatisticalOutlierRemovalOperation(nb_neighbors, std_ratio))
        return self
    
    def radius_outlier_removal(self, nb_points: int, radius: float):
        self.steps.append(RadiusOutlierRemovalOperation(nb_points, radius))
        return self
    
    @property
    def summary(self):
        return '\n'.join(['{}. '.format(idx + 1) + str(step) for idx, step in enumerate(self.steps)])
    
    def _run(self):
        print('Steps:')
        print(self.summary)
        
        print('Processing {} PCD files...'.format(len(self.pcd_files)))
        for pcd_file in progressbar.progressbar(self.pcd_files, redirect_stdout=True):
            pcd_file_base = os.path.basename(pcd_file)
            pcd_file_out = os.path.join(self.out_dir, pcd_file_base)
            
            pcd = o3d.io.read_point_cloud(pcd_file)
            
            for step in self.steps:
                pcd = step.run(pcd)
                
            o3d.io.write_point_cloud(pcd_file_out, pcd)
            del pcd
            
    
    def __str__(self) -> str:
        self._run()
        return 'Done!'


if __name__ == '__main__':
    fire.Fire(BatchPointCloudProcessor)
