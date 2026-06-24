import numpy as np
from scipy.spatial import KDTree
from Bio.SeqUtils.ProtParam import ProteinAnalysis

class PocketDetector:
    """高性能蛋白质成药口袋探测引擎"""
    
    @staticmethod
    def detect(valid_atoms, coords, grid_res=2.0, min_dist=1.5, max_dist=4.0):
        """
        基于网格法探测蛋白质口袋特征
        """
        min_bound = np.min(coords, axis=0) - 5.0
        max_bound = np.max(coords, axis=0) + 5.0
        
        x = np.arange(min_bound[0], max_bound[0], grid_res)
        y = np.arange(min_bound[1], max_bound[1], grid_res)
        z = np.arange(min_bound[2], max_bound[2], grid_res)
        xv, yv, zv = np.meshgrid(x, y, z)
        grid_points = np.c_[xv.ravel(), yv.ravel(), zv.ravel()]
        
        tree = KDTree(coords)
        distances, _ = tree.query(grid_points, k=1)
        pocket_points = grid_points[(distances > min_dist) & (distances < max_dist)]
        
        pocket_center = np.mean(pocket_points, axis=0) if len(pocket_points) > 0 else np.mean(coords, axis=0)
        pocket_volume = len(pocket_points) * (grid_res ** 3)
        
        # 疏水性分析
        indices = tree.query_ball_point(pocket_center, r=5.0)
        nearby_residues = valid_atoms.iloc[indices]['label_comp_id'].unique()
        
        aa_map = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E', 
                  'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 
                  'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}
        seq_1letter = "".join([aa_map.get(res, '') for res in nearby_residues])
        
        hydrophobicity = 0.0
        if seq_1letter:
            hydrophobicity = ProteinAnalysis(seq_1letter).gravy()
            
        return {
            "center": pocket_center,
            "volume": pocket_volume,
            "hydrophobicity": hydrophobicity,
            "residues": nearby_residues,
            "pocket_points": pocket_points  # 口袋网格点坐标，用于 3D 形状可视化
        }