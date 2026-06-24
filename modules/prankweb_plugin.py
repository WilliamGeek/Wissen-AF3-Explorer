import requests
import gc
import sys
import time
import json
import tempfile
import os


class PrankWebRunner:
    """PrankWeb 蛋白口袋预测全自动运行

    通过 PrankWeb REST API 提交结构并获取口袋预测结果。
    PrankWeb 基于 P2Rank 机器学习算法，提供文档化的公开 API。
    API: https://prankweb.cz/api/v2/
    """

    API_BASE = "https://prankweb.cz/api/v2"

    def __init__(self, structure_path: str, job_name: str = "unknown"):
        self.structure_path = structure_path
        self.job_name = job_name
        self.prediction_id = ""
        self.pockets = []
        self.error = None
        self._tmp_pdb = None

    def run(self) -> bool:
        """执行 PrankWeb 口袋预测"""
        # Step 1: 准备结构文件 (PrankWeb 接受 PDB 和 mmCIF)
        structure_file = self._prepare_structure()
        if not structure_file:
            return False

        print(f"[PrankWeb] 提交结构: {os.path.basename(structure_file)}", file=sys.stderr)

        # Step 2: 提交预测任务
        if not self._submit_prediction(structure_file):
            return False

        # Step 3: 轮询结果
        if not self._poll_results():
            return False

        print(f"[PrankWeb] 预测完成: {len(self.pockets)} 个口袋", file=sys.stderr)
        return True

    def _prepare_structure(self) -> str:
        """准备结构文件，CIF 需转换为 PDB"""
        if self.structure_path.endswith((".pdb", ".PDB")):
            return self.structure_path

        # CIF → PDB 转换
        if self.structure_path.endswith((".cif", ".cif.gz")):
            try:
                from modules.foldseek_plugin import FoldseekAPIWrapper
                wrapper = FoldseekAPIWrapper(self.structure_path)
                pdb_content, _ = wrapper._read_file(self.structure_path)
                del wrapper; gc.collect()

                # 写入临时 PDB 文件
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".pdb", delete=False, mode="w", encoding="utf-8"
                )
                tmp.write(pdb_content)
                tmp.close()
                self._tmp_pdb = tmp.name
                print(f"[PrankWeb] CIF 已转换为临时 PDB: {tmp.name}", file=sys.stderr)
                return tmp.name
            except Exception as e:
                self.error = f"CIF→PDB 转换失败: {str(e)}"
                return ""

        self.error = f"不支持的结构文件格式: {self.structure_path}"
        return ""

    def _submit_prediction(self, structure_file: str) -> bool:
        """提交结构到 PrankWeb API"""
        try:
            url = f"{self.API_BASE}/prediction/v4-user-upload"

            # 构造 configuration.json
            config = {
                "chains": [],
                "structure-sealed": True,
                "prediction-model": "conservation_hmm",
            }

            with open(structure_file, "rb") as f:
                files = {
                    "structure": (os.path.basename(structure_file), f, "chemical/x-pdb"),
                    "configuration": ("configuration.json", json.dumps(config), "application/json"),
                }
                resp = requests.post(url, files=files, timeout=60)

            if resp.status_code not in (200, 201):
                self.error = f"PrankWeb 提交失败: HTTP {resp.status_code}\n{resp.text[:500]}"
                return False

            data = resp.json()
            self.prediction_id = data.get("id", "")
            if not self.prediction_id:
                self.error = f"PrankWeb 未返回任务 ID: {data}"
                return False

            print(f"[PrankWeb] 任务已提交: {self.prediction_id}", file=sys.stderr)
            return True

        except requests.exceptions.Timeout:
            self.error = "PrankWeb API 提交超时。"
            return False
        except Exception as e:
            self.error = f"PrankWeb 提交异常: {str(e)}"
            return False

    def _poll_results(self) -> bool:
        """轮询 PrankWeb 预测结果"""
        url = f"{self.API_BASE}/prediction/v4-user-upload/{self.prediction_id}"
        max_retries = 120  # 最多等 4 分钟

        for attempt in range(max_retries):
            time.sleep(2)
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                data = resp.json()
                status = str(data.get("status", "")).upper()

                if attempt % 10 == 0:
                    print(f"[PrankWeb] 轮询 {attempt + 1}: status={status}", file=sys.stderr)

                if status == "SUCCESS" or status == "FINISHED":
                    return self._parse_results(data)
                elif status in ("ERROR", "FAILED"):
                    self.error = f"PrankWeb 任务失败: {status}"
                    return False
                # PENDING / RUNNING 继续等待

            except Exception as e:
                print(f"[PrankWeb] 轮询异常: {e}", file=sys.stderr)
                continue

        self.error = "PrankWeb 轮询超时（>4分钟）。"
        return False

    def _parse_results(self, data: dict) -> bool:
        """解析 PrankWeb 返回的口袋预测结果"""
        try:
            # PrankWeb 返回结构包含 pockets 列表
            pockets_raw = data.get("pockets", [])
            if not pockets_raw:
                # 尝试从 predictions 字段获取
                pockets_raw = data.get("predictions", data.get("results", []))

            for p in pockets_raw:
                self.pockets.append({
                    "rank": p.get("rank", p.get("id", 0)),
                    "score": round(p.get("score", p.get("probability", 0)), 4),
                    "center_x": round(p.get("center", [0, 0, 0])[0], 2) if isinstance(p.get("center"), list) else p.get("centerX", 0),
                    "center_y": round(p.get("center", [0, 0, 0])[1], 2) if isinstance(p.get("center"), list) else p.get("centerY", 0),
                    "center_z": round(p.get("center", [0, 0, 0])[2], 2) if isinstance(p.get("center"), list) else p.get("centerZ", 0),
                    "residues": p.get("residues", p.get("surroundingResidues", [])),
                    "surface": p.get("surface", p.get("surfaceAa", "")),
                })

            if not self.pockets:
                # 如果没有 pockets 字段，保存原始数据
                self.pockets = [{"raw": data}]

            return True

        except Exception as e:
            self.error = f"PrankWeb 结果解析异常: {str(e)}"
            return False

    def get_results_df_data(self) -> list:
        """返回适合 DataFrame 的口袋数据"""
        return [
            {
                "Rank": p.get("rank", i + 1),
                "Score": p.get("score", 0),
                "Center": f"({p.get('center_x', 0)}, {p.get('center_y', 0)}, {p.get('center_z', 0)})",
                "Residues": len(p.get("residues", [])) if isinstance(p.get("residues"), list) else str(p.get("residues", ""))[:50],
            }
            for i, p in enumerate(self.pockets)
            if "raw" not in p
        ]

    def cleanup(self):
        """清理临时文件"""
        if self._tmp_pdb and os.path.exists(self._tmp_pdb):
            try:
                os.unlink(self._tmp_pdb)
            except Exception:
                pass
        self.pockets = []
        gc.collect()
