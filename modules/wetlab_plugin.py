import gc
import sys
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from utils.biopandas_patch import safe_read_mmcif

class BiochemProfiler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.sequence = ""
        self.metrics = {}
        self.error = None
        self._extract_and_calculate()

    def _extract_and_calculate(self):
        try:
            # 1. 安全读取 CIF（兼容 AF3 缺失列）
            pmmcif = safe_read_mmcif(self.file_path)
            
            # 尝试从实体多聚物信息中获取标准序列
            try:
                seq_df = pmmcif.df['entity_poly']
                if not seq_df.empty and 'pdbx_seq_one_letter_code' in seq_df.columns:
                    raw_seq = seq_df['pdbx_seq_one_letter_code'].iloc[0]
                    self.sequence = raw_seq.replace('\n', '').replace(' ', '')
            except KeyError:
                pass
            
            # 若上述字段缺失，降级基于 CA 原子的残基序列提取
            if not self.sequence:
                try:
                    atom_df = pmmcif.df['ATOM']
                    
                    # 动态判定列名，严格兼容 mmCIF 与常规 PDB 格式
                    atom_col = 'label_atom_id' if 'label_atom_id' in atom_df.columns else 'atom_name'
                    comp_col = 'label_comp_id' if 'label_comp_id' in atom_df.columns else 'residue_name'
                    
                    ca_df = atom_df[atom_df[atom_col] == 'CA']
                    
                    aa_map = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E', 
                              'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 
                              'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}
                    
                    seq_list = [aa_map.get(res, 'X') for res in ca_df[comp_col]]
                    self.sequence = "".join(seq_list).replace('X', '')
                except Exception as ex:
                    print(f"提取后备序列时发生异常: {str(ex)}", file=sys.stderr)
            
            # 极致内存回收：必须在此处摧毁大体积的 DataFrame
            if 'atom_df' in locals(): del atom_df
            if 'ca_df' in locals(): del ca_df
            del pmmcif
            gc.collect()
            
            if not self.sequence:
                self.error = "未能成功从 CIF 文件中提取有效序列。"
                return
            
            # 2. 理化性质运算
            analysed_seq = ProteinAnalysis(self.sequence)
            self.metrics['MW'] = analysed_seq.molecular_weight()
            self.metrics['pI'] = analysed_seq.isoelectric_point()
            self.metrics['Instability_Index'] = analysed_seq.instability_index()
            
        except Exception as e:
            self.error = f"解析结构与计算理化性质时发生错误: {str(e)}"
            print(f"详细错误堆栈: {str(e)}", file=sys.stderr)
            gc.collect()