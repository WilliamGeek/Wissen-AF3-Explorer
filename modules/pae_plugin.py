import os
import re
import json
import gc
import glob

class PaeVisualizer:
    def __init__(self, cif_path):
        self.cif_path = cif_path
        self.pae_matrix = None
        self.error = None
        self._find_and_load_pae()

    def _find_and_load_pae(self):
        dir_name = os.path.dirname(self.cif_path)
        base_name = os.path.basename(self.cif_path)

        # 去除扩展名
        if base_name.endswith('.cif.gz'):
            name_without_ext = base_name[:-7]
        elif base_name.endswith('.cif'):
            name_without_ext = base_name[:-4]
        else:
            name_without_ext = os.path.splitext(base_name)[0]

        # 提取 model 编号 (如 fold_xxx_model_0.cif -> model_num=0)
        model_num = None
        m = re.search(r'model[_\-](\d+)', base_name)
        if m:
            model_num = int(m.group(1))

        # 构建候选文件名列表（按优先级）
        possible_names = [
            f"{name_without_ext}.json",
            f"{name_without_ext}_full_data.json",
            f"{name_without_ext}_pae.json",
            "pae.json"
        ]
        # 若提取到 model 编号，优先匹配对应的 full_data_N.json
        if model_num is not None:
            possible_names.insert(0, f"{name_without_ext.rsplit('_model', 1)[0]}_full_data_{model_num}.json")
            # 也尝试替代 naming pattern (如 fold_xxx_full_data_0.json)
            possible_names.insert(0, f"*_full_data_{model_num}.json")

        json_path = None
        for j_name in possible_names:
            # 支持 glob 通配符
            if '*' in j_name or '?' in j_name:
                matches = glob.glob(os.path.join(dir_name, j_name))
                if matches:
                    json_path = matches[0]
                    break
            else:
                p = os.path.join(dir_name, j_name)
                if os.path.exists(p):
                    json_path = p
                    break

        # 最后的兜底：查找目录下任意包含 pae 或 full_data 的 json 文件
        if not json_path:
            for pattern in ['*pae*.json', '*full_data*.json', '*.json']:
                candidates = glob.glob(os.path.join(dir_name, pattern))
                # 过滤掉 confidences 文件
                candidates = [c for c in candidates if 'confidence' not in os.path.basename(c).lower()]
                if candidates:
                    json_path = candidates[0]
                    break

        if not json_path:
            self.error = (
                "未检测到对应的 PAE JSON 文件。\n"
                "AlphaFold 3 的 PAE 数据通常存储在 *_full_data_*.json 文件中，\n"
                "请确保完整解压 AF3 输出压缩包后使用本工具。"
            )
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 兼容多种 JSON 数据结构
            if isinstance(data, dict):
                if "pae" in data:
                    self.pae_matrix = data["pae"]
                elif "predicted_aligned_error" in data:
                    self.pae_matrix = data["predicted_aligned_error"]
                else:
                    # 尝试递归查找
                    found = self._find_pae_in_dict(data)
                    if found is not None:
                        self.pae_matrix = found
                    else:
                        self.error = f"JSON 文件 ({os.path.basename(json_path)}) 中未找到 'pae' 数据。可用键: {list(data.keys())[:10]}"
            elif isinstance(data, list):
                # job_request.json 是 list 格式
                for item in data:
                    if isinstance(item, dict):
                        found = self._find_pae_in_dict(item)
                        if found is not None:
                            self.pae_matrix = found
                            break
                if self.pae_matrix is None:
                    self.error = "JSON 文件列表中未找到 'pae' 数据"
            else:
                self.error = "JSON 格式无法解析或未包含 PAE 数据"

            del data
            gc.collect()

        except Exception as e:
            self.error = f"解析 JSON 文件时出错: {e}"
            gc.collect()

    def _find_pae_in_dict(self, d, depth=0):
        """递归在嵌套字典中查找 pae 键"""
        if depth > 3:
            return None
        if "pae" in d:
            return d["pae"]
        if "predicted_aligned_error" in d:
            return d["predicted_aligned_error"]
        for v in d.values():
            if isinstance(v, dict):
                found = self._find_pae_in_dict(v, depth + 1)
                if found is not None:
                    return found
        return None