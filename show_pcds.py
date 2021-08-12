import glob
import os
import time

import open3d as o3d
import fire
import progressbar


def show_pcd_video(pcd_dir: str, fps: float = 10.0):
    pcd_files = glob.glob(os.path.join(pcd_dir, '*.pcd'))
    pcd_files.sort(key=lambda v: int(os.path.basename(v).split('_')[3]))
    
    pcds = []
    print('Loading PCD files...')
    for pcd_file in progressbar.progressbar(pcd_files, redirect_stdout=True):
        pcd = o3d.io.read_point_cloud(pcd_file)
        print('{:<50} {:>20} points'.format(pcd_file, len(pcd.points)))
        pcds.append(pcd)

    # Show animation
    print('Showing animation... (Esc to exit)')
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(pcds[0])

    dt = 1/fps
    t = time.time()

    idx = 0
    while vis.poll_events():
        if time.time() - t >= dt:
            vis.remove_geometry(pcds[idx], reset_bounding_box=False)
        
            idx += 1
            idx %= len(pcds)
        
            vis.add_geometry(pcds[idx], reset_bounding_box=False)
            
            t = time.time()
        
        vis.update_renderer()
    
    vis.destroy_window()
    # print('Open3D\'s visualization is poorly managed, so you may need to do Ctrl+C to exit this program.')
        

if __name__ == '__main__':
    fire.Fire(show_pcd_video)
