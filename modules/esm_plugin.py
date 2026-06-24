import requests
import gc
import sys
import re


class ESMFoldRunner:
    """ESM Atlas / ESMFold 全自动结构预测

    通过 ESM Atlas API 直接折叠蛋白序列，无需手动访问网站。
    API: https://api.esmatlas.com/foldSequence/v1/pdb/
    """

    API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

    def __init__(self, sequence: str, job_name: str = "unknown"):
        self.sequence = sequence
        self.job_name = job_name
        self.pdb_content = ""
        self.plddt_scores = []
        self.mean_plddt = 0.0
        self.ptm = None
        self.residue_count = 0
        self.error = None

    def run(self) -> bool:
        """调用 ESMFold API 进行结构预测"""
        if not self.sequence:
            self.error = "无有效序列，无法进行 ESMFold 预测。"
            return False

        seq_len = len(self.sequence)
        print(f"[ESMFold] 序列长度: {seq_len} aa，开始 API 折叠...", file=sys.stderr)

        # ESM Atlas API 对序列长度有限制（通常 < 400 aa 效果最佳，但支持更长）
        if seq_len > 1000:
            print(f"[ESMFold] 警告: 序列较长 ({seq_len} aa)，API 可能超时或拒绝", file=sys.stderr)

        try:
            resp = requests.post(
                self.API_URL,
                data=self.sequence,
                headers={"Content-Type": "text/plain"},
                timeout=300  # 长序列可能需要较长时间
            )

            if resp.status_code != 200:
                self.error = (
                    f"ESM Atlas API 返回 HTTP {resp.status_code}\n"
                    f"响应: {resp.text[:500]}\n\n"
                    "可能原因:\n"
                    "1) 序列过长，API 拒绝处理\n"
                    "2) API 速率限制，请稍后重试\n"
                    "3) 服务端暂时不可用"
                )
                return False

            self.pdb_content = resp.text

            # 验证返回内容是否为有效 PDB
            if not self.pdb_content.strip().startswith(("ATOM", "HEADER", "TITLE", "MODEL")):
                self.error = f"API 返回内容非有效 PDB 格式:\n{self.pdb_content[:300]}"
                return False

            # 解析 PDB 中的 pLDDT 和 pTM
            self._parse_pdb_confidence()

            print(
                f"[ESMFold] 预测完成: {self.residue_count} 残基, "
                f"平均 pLDDT={self.mean_plddt:.2f}",
                file=sys.stderr
            )
            return True

        except requests.exceptions.Timeout:
            self.error = "ESM Atlas API 请求超时（>300s），序列可能过长或服务器繁忙。"
            return False
        except requests.exceptions.ConnectionError:
            self.error = "无法连接 ESM Atlas API，请检查网络连接。"
            return False
        except Exception as e:
            self.error = f"ESMFold 调用异常: {str(e)}"
            return False

    def _parse_pdb_confidence(self):
        """从 ESMFold PDB 输出中提取 pLDDT 和 pTM"""
        plddt_values = []
        residue_plddt = {}  # res_seq -> pLDDT

        for line in self.pdb_content.splitlines():
            if line.startswith("ATOM"):
                try:
                    # PDB 固定列格式: B-factor 在 61-66 列
                    bfactor = float(line[60:66].strip())
                    res_seq = int(line[22:26].strip())
                    residue_plddt[res_seq] = bfactor
                except (ValueError, IndexError):
                    continue
            elif line.startswith("REMARK") and "pTM" in line:
                # 尝试从 REMARK 行提取 pTM
                match = re.search(r'pTM[:\s]*([0-9.]+)', line, re.IGNORECASE)
                if match:
                    self.ptm = float(match.group(1))

        if residue_plddt:
            self.plddt_scores = list(residue_plddt.values())
            self.mean_plddt = sum(self.plddt_scores) / len(self.plddt_scores)
            self.residue_count = len(residue_plddt)

    def get_confidence_summary(self) -> dict:
        """返回置信度分级统计"""
        if not self.plddt_scores:
            return {}

        bins = {">90": 0, "70-90": 0, "50-70": 0, "<50": 0}
        for v in self.plddt_scores:
            if v > 90:
                bins[">90"] += 1
            elif v >= 70:
                bins["70-90"] += 1
            elif v >= 50:
                bins["50-70"] += 1
            else:
                bins["<50"] += 1

        return {
            "mean_plddt": round(self.mean_plddt, 2),
            "ptm": self.ptm,
            "residue_count": self.residue_count,
            "distribution": bins,
        }

    def get_pdb_download(self) -> tuple:
        """返回 (pdb_content, filename) 供下载"""
        return self.pdb_content, f"{self.job_name}_esmfold.pdb"

    def cleanup(self):
        self.pdb_content = ""
        self.plddt_scores = []
        gc.collect()
