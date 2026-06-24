import os
import zipfile
import json
import shutil
import uuid
from pathlib import Path
import pandas as pd

class AF3ValidationError(Exception):
    """自定义异常，用于处理非标准 AlphaFold 3 输出压缩包或解析错误"""
    pass

class AF3ZipParser:
    """AlphaFold 3 输出 ZIP 压缩包智能解析引擎"""
    def __init__(self, base_temp_dir: str = "temp_workspace"):
        self.base_temp_dir = Path(base_temp_dir)
        self.current_workspace = None
        self.data_json = None
        self.ranking_df = None

    def parse(self, zip_file) -> tuple[dict, pd.DataFrame]:
        """
        解析 Streamlit 上传的 .zip 文件流
        :param zip_file: Streamlit UploadedFile 对象或类文件流对象
        :return: 元组 (data_json_dict, ranking_df)
        :raises AF3ValidationError: 当文件格式不符、缺失核心数据文件或解析失败时抛出
        """
        try:
            self.base_temp_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise AF3ValidationError(f"创建临时工作空间根目录失败: {str(e)}")

        self.current_workspace = self.base_temp_dir / f"af3_run_{uuid.uuid4().hex}"
        self.current_workspace.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.current_workspace)
        except Exception as e:
            self.cleanup()
            raise AF3ValidationError(f"压缩包解压失败，请检查文件是否为合法的 ZIP 压缩包或是否损坏。详情: {str(e)}")

        # 深度扫描解压目录树，检索核心数据文件
        data_files = []
        conf_files = []
        all_files = []

        for root, _, files in os.walk(self.current_workspace):
            for file in files:
                all_files.append(file)
                f_lower = file.lower()
                # 兼容性匹配：包含 data/ranking/confidences 关键词
                if "data" in f_lower and file.endswith(".json"):
                    data_files.append(Path(root) / file)
                elif "confidences" in f_lower and file.endswith(".json"):
                    conf_files.append(Path(root) / file)

        if not data_files:
            self.cleanup()
            file_list_str = ", ".join(all_files[:20])
            raise AF3ValidationError(f"非标准的 AlphaFold 3 输出目录！未找到 data JSON 文件。检测到: [{file_list_str}...]。")

        # 读取数据
        try:
            with open(data_files[0], 'r', encoding='utf-8') as f:
                self.data_json = json.load(f)
        except Exception as e:
            self.cleanup()
            raise AF3ValidationError(f"读取参数档案失败: {str(e)}")

        # 动态构建评分排行榜
        rankings = []
        for cf in conf_files:
            try:
                with open(cf, 'r', encoding='utf-8') as f:
                    c_data = json.load(f)
                    rankings.append({
                        "model_name": cf.name,
                        "ranking_score": c_data.get("ptm", 0.0)
                    })
            except:
                continue
        
        self.ranking_df = pd.DataFrame(rankings).sort_values(by="ranking_score", ascending=False).reset_index(drop=True) if rankings else pd.DataFrame(columns=["model_name", "ranking_score"])
        
        return self.data_json, self.ranking_df

    def get_best_cif_path(self) -> str:
        """获取 AlphaFold 3 预测排名第一的最佳结构 CIF 文件的绝对路径
        
        支持在解压目录树中深度搜索 CIF 文件（兼容 AF3 输出子目录结构）。
        """
        if not self.current_workspace or not self.current_workspace.exists():
            raise AF3ValidationError("无法获取模型路径，请先成功上传并解析合法的 AlphaFold 3 压缩包。")

        # 深度递归搜索 CIF 文件（兼容子目录结构）
        cif_files = []
        for root, _, files in os.walk(self.current_workspace):
            for file in files:
                if file.endswith(".cif"):
                    cif_files.append(Path(root) / file)
        
        if not cif_files:
            raise AF3ValidationError("在解压目录树中未检索到代表最优结构的 `.cif` 文件！")
        
        # 优先选择 model_0 或包含 "model" 关键字的文件作为最佳结构
        for cf in cif_files:
            fname = cf.name.lower()
            if "model_0" in fname or "ranked_0" in fname:
                return str(cf.resolve())
        
        return str(cif_files[0].resolve())

    def extract_confidences(self) -> dict:
        """从 confidences.json 提取浓缩特征"""
        if not self.current_workspace or not self.current_workspace.exists():
            raise AF3ValidationError("无法提取置信度，请先成功上传并解析合法的 AlphaFold 3 压缩包。")

        conf_path = None
        for root, _, files in os.walk(self.current_workspace):
            for file in files:
                if "confidences" in file.lower() and file.endswith(".json"):
                    conf_path = Path(root) / file
                    break
            if conf_path: break

        if not conf_path:
            raise AF3ValidationError("未在压缩包中检索到置信度配置文件 (`_confidences.json`)。")

        with open(conf_path, 'r', encoding='utf-8') as f:
            conf_data = json.load(f)

        plddt_list = conf_data.get("plddt", [])
        mean_plddt = sum(plddt_list) / len(plddt_list) if plddt_list and isinstance(plddt_list, list) else 0.0

        return {
            "job_name": self.data_json.get("name", "未命名任务") if self.data_json else "未命名任务",
            "job_id": self.data_json.get("id", "N/A") if self.data_json else "N/A",
            "ptm": conf_data.get("ptm", "N/A"),
            "iptm": conf_data.get("iptm", "N/A"),
            "mean_plddt": round(mean_plddt, 4),
            "fraction_confidently_predicted": conf_data.get("fraction_confidently_predicted", "N/A"),
            "model_type": self.data_json.get("model", {}).get("model_type", "N/A") if self.data_json else "N/A"
        }

    def cleanup(self):
        """强力物理删除当前生成的临时工作子目录"""
        if self.current_workspace and self.current_workspace.exists():
            try: shutil.rmtree(self.current_workspace)
            except: pass
            finally: self.current_workspace = None