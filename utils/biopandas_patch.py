"""
biopandas 兼容性补丁 - 处理 AlphaFold 3 CIF 文件中缺失列的问题

背景：biopandas 内置的 mmcif_col_types 字典假设所有标准 mmCIF 列都存在，
但 AlphaFold 3 输出的 CIF 文件不包含某些可选列（如 auth_atom_id, pdbx_formal_charge 等），
导致 pandas astype() 时抛出 KeyError。

策略：捕获 KeyError，动态移除报错的列名，重试读取，最多重试 10 次。
"""
import biopandas.mmcif.pandas_mmcif as _bmm
from biopandas.mmcif import PandasMmcif as _PandasMmcif


def safe_read_mmcif(cif_path: str, max_retries: int = 10) -> _PandasMmcif:
    """
    安全读取 mmCIF 文件，自动处理 AF3 输出文件中不存在的列。
    
    遇到 pandas KeyError "column not found in columns" 时，
    自动从 biopandas 的 dtype 映射中移除该列后重试。
    
    :param cif_path: CIF 文件路径
    :param max_retries: 最大重试次数
    :return: PandasMmcif 实例
    """
    # 保存原始映射的副本用于恢复
    original = dict(_bmm.mmcif_col_types)
    
    try:
        for attempt in range(max_retries):
            try:
                result = _PandasMmcif().read_mmcif(cif_path)
                return result
            except KeyError as e:
                # 提取缺失的列名: "Only a column name can be used... 'col_name' not found..."
                error_msg = str(e)
                col_name = _extract_missing_column(error_msg)
                if col_name and col_name in _bmm.mmcif_col_types:
                    del _bmm.mmcif_col_types[col_name]
                elif col_name:
                    # 列名已不在映射中但仍然报错，尝试原始读取
                    pass
                else:
                    # 无法解析列名，重新抛出
                    raise
    
        # 超过最大重试次数
        raise RuntimeError(f"safe_read_mmcif: 超过最大重试次数 ({max_retries})，仍无法读取 {cif_path}")
    finally:
        # 始终恢复原始映射，防止模块级状态污染
        _bmm.mmcif_col_types.clear()
        _bmm.mmcif_col_types.update(original)


def _extract_missing_column(error_msg: str):
    """从 KeyError 消息中提取缺失的列名"""
    import re
    # 匹配 "... 'col_name' not found ..."
    match = re.search(r"'([^']+)' not found in columns", error_msg)
    if match:
        return match.group(1)
    # 备用: 匹配最后一个被引号包裹的词
    matches = re.findall(r"'([^']+)'", error_msg)
    return matches[0] if matches else None