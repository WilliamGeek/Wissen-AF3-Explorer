import gc
import pandas as pd
from utils.biopandas_patch import safe_read_mmcif

class CifParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.metadata = {}
        self.plddt_stats = {}
        self.error = None
        self._parse()

    def _parse(self):
        try:
            # 安全读取 CIF（兼容 AF3 缺失列）
            pmmcif = safe_read_mmcif(self.file_path)
            
            # Extract ATOM data
            atom_df = pmmcif.df['ATOM']
            
            if atom_df.empty:
                del pmmcif
                gc.collect()
                return

            # Extract C-alpha atoms to represent residues
            # 动态判定列名: AF3 CIF 使用 label_atom_id，旧版 mmCIF 使用 atom_name
            atom_col = 'label_atom_id' if 'label_atom_id' in atom_df.columns else 'atom_name'
            ca_df = atom_df[atom_df[atom_col] == 'CA']
            
            # 1. Metadata extraction
            self.metadata['length'] = len(ca_df)
            self.metadata['entity_count'] = atom_df['label_entity_id'].nunique()
            self.metadata['protein_name'] = "AlphaFold Model"

            # 2. Quality data (pLDDT) extraction and classification
            if 'B_iso_or_equiv' in ca_df.columns:
                plddt = ca_df['B_iso_or_equiv'].astype(float)
                self.plddt_stats = {
                    '>90': int((plddt > 90).sum()),
                    '70-90': int(((plddt > 70) & (plddt <= 90)).sum()),
                    '50-70': int(((plddt > 50) & (plddt <= 70)).sum()),
                    '<50': int((plddt <= 50).sum())
                }
                del plddt
            else:
                self.plddt_stats = {}
                
            # Memory management
            del ca_df
            del atom_df
            del pmmcif
            gc.collect()
        except Exception as e:
            self.error = f"解析 CIF 文件时发生错误: {str(e)}"
            print(f"Error parsing CIF: {e}")
            gc.collect()