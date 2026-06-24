import gc
import numpy as np
import py3Dmol

def render_protein(cif_path: str):
    """
    高性能读取并构建 AlphaFold 3 最优模型的 3D 交互式视图
    
    :param cif_path: .cif 文件的绝对物理路径
    :return: py3Dmol.view 渲染视图实例
    """
    with open(cif_path, "r", encoding="utf-8") as f:
        cif_string = f.read()
        
    view = py3Dmol.view(width=800, height=600)
    view.addModel(cif_string, "cif")
    
    # 使用 pLDDT 置信度渐变色 (蓝=高置信, 红=低置信)
    view.setStyle({'cartoon': {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 50, 'max': 100}}})
    view.setBackgroundColor('#0e1117')
    
    # 自动居中并缩放
    view.zoomTo()
    view.center()
    
    del cif_string
    gc.collect()
    
    return view


def render_protein_with_pocket(cif_path: str, pocket_center, pocket_radius=4.0, pocket_color='magenta',
                                pocket_points=None, point_radius=0.3):
    """
    渲染蛋白质 3D 结构并在指定位置叠加口袋球体标记
    
    :param cif_path: CIF 文件路径
    :param pocket_center: 口袋中心坐标 (x, y, z) 或 numpy array
    :param pocket_radius: 中心球体半径 (Å)
    :param pocket_color: 球体颜色
    :param pocket_points: 口袋网格点坐标数组 (N, 3)，用于展示口袋大致形状
    :param point_radius: 单个网格点缀球半径 (Å)
    :return: py3Dmol.view
    """
    with open(cif_path, "r", encoding="utf-8") as f:
        cif_string = f.read()
    
    view = py3Dmol.view(width=800, height=600)
    view.addModel(cif_string, "cif")
    view.setStyle({'cartoon': {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 50, 'max': 100}}})
    view.setBackgroundColor('#0e1117')
    
    # ── 渲染口袋形状网格点 ──
    if pocket_points is not None and len(pocket_points) > 0:
        # 限制最多渲染 2000 个点以避免浏览器卡顿
        pts = np.array(pocket_points)
        if len(pts) > 2000:
            # 均匀采样降采样
            indices = np.linspace(0, len(pts) - 1, 2000, dtype=int)
            pts = pts[indices]
        
        # 构建多球体的 xyz 样式字符串或逐个添加
        for pt in pts:
            view.addSphere({
                'center': {'x': float(pt[0]), 'y': float(pt[1]), 'z': float(pt[2])},
                'radius': point_radius,
                'color': pocket_color,
                'alpha': 0.35
            })
    
    # ── 叠加口袋中心大球体标记 ──
    cx, cy, cz = float(pocket_center[0]), float(pocket_center[1]), float(pocket_center[2])
    view.addSphere({
        'center': {'x': cx, 'y': cy, 'z': cz},
        'radius': pocket_radius,
        'color': pocket_color,
        'alpha': 0.5
    })
    
    view.zoomTo()
    view.center()
    
    del cif_string
    gc.collect()
    
    return view