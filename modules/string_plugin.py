import requests
import gc
import sys
import time


class STRINGQuery:
    """STRING 蛋白互作网络全自动查询

    通过 STRING REST API 直接获取蛋白-蛋白互作数据。
    API: https://string-db.org/api
    无需认证，支持 TSV/JSON/XML/PNG/SVG 输出。
    """

    API_BASE = "https://string-db.org/api"
    HUMAN_TAXON = 9606  # 人类

    def __init__(self, identifier: str, species: int = HUMAN_TAXON):
        self.identifier = identifier.strip()
        self.species = species
        self.string_id = ""
        self.preferred_name = ""
        self.interactions = []
        self.network_image_url = ""
        self.error = None

    def run(self, required_score: int = 400, add_nodes: int = 10) -> bool:
        """执行 STRING 查询

        Args:
            required_score: 置信度阈值 (150=低, 400=中, 700=高, 900=最高)
            add_nodes: 额外添加的互作节点数 (0-10)
        """
        if not self.identifier:
            self.error = "无有效蛋白标识符。"
            return False

        print(f"[STRING] 查询: {self.identifier}, species={self.species}", file=sys.stderr)

        # Step 1: 映射标识符到 STRING ID
        if not self._resolve_identifier():
            return False

        # Step 2: 获取互作网络
        if not self._get_interactions(required_score, add_nodes):
            return False

        # Step 3: 生成网络图 URL
        self.network_image_url = (
            f"{self.API_BASE}/image/network?"
            f"identifiers={self.string_id}&species={self.species}"
            f"&required_score={required_score}&network_type=functional"
            f"&network_flavor=evidence"
        )

        print(f"[STRING] 找到 {len(self.interactions)} 条互作关系", file=sys.stderr)
        return True

    def _resolve_identifier(self) -> bool:
        """将蛋白名/UniProt ID 映射为 STRING ID"""
        try:
            url = f"{self.API_BASE}/tsv/get_string_ids"
            params = {
                "identifiers": self.identifier,
                "species": self.species,
                "echo_query": 1,
                "limit": 1,
                "caller_identity": "Wissen_AF3_Explorer",
            }
            resp = requests.get(url, params=params, timeout=30)

            if resp.status_code != 200:
                if resp.status_code == 400:
                    self.error = (
                        f"STRING 无法识别标识符 '{self.identifier}'。\n"
                        "请检查:\n"
                        "1) 蛋白名称拼写是否正确（如 TP53、EGFR、BRCA1）\n"
                        "2) UniProt ID 格式是否正确（如 P04637）\n"
                        "3) 该蛋白是否属于人类 (taxid=9606)"
                    )
                else:
                    self.error = f"STRING 标识符映射失败: HTTP {resp.status_code}"
                return False

            lines = resp.text.strip().split("\n")
            if len(lines) < 2:
                self.error = (
                    f"STRING 未找到 '{self.identifier}' 的匹配。\n"
                    "可能原因:\n"
                    "1) 蛋白名称拼写错误\n"
                    "2) 该物种不在 STRING 数据库中\n"
                    "3) UniProt ID 需要带物种前缀"
                )
                return False

            # 解析 TSV: queryItem, queryIndex, stringId, ncbiTaxonId, taxonName, preferredName, annotation
            fields = lines[1].split("\t")
            if len(fields) >= 6:
                self.string_id = fields[2]
                self.preferred_name = fields[5]
                print(f"[STRING] 映射成功: {self.identifier} -> {self.string_id} ({self.preferred_name})",
                      file=sys.stderr)
                return True
            else:
                self.error = f"STRING 返回格式异常: {lines[1][:200]}"
                return False

        except requests.exceptions.Timeout:
            self.error = "STRING API 请求超时。"
            return False
        except Exception as e:
            self.error = f"STRING 标识符映射异常: {str(e)}"
            return False

    def _get_interactions(self, required_score: int, add_nodes: int) -> bool:
        """获取互作网络数据"""
        try:
            url = f"{self.API_BASE}/tsv/network"
            params = {
                "identifiers": self.string_id,
                "species": self.species,
                "required_score": required_score,
                "network_type": "functional",
                "add_nodes": add_nodes,
                "caller_identity": "Wissen_AF3_Explorer",
            }
            resp = requests.get(url, params=params, timeout=60)

            if resp.status_code != 200:
                self.error = f"STRING 网络查询失败: HTTP {resp.status_code}"
                return False

            lines = resp.text.strip().split("\n")
            if len(lines) < 2:
                self.interactions = []
                return True

            # 解析 TSV 表头
            headers = lines[0].split("\t")

            for line in lines[1:]:
                fields = line.split("\t")
                if len(fields) >= len(headers):
                    row = dict(zip(headers, fields))
                    self.interactions.append({
                        "protein_a": row.get("preferredName_A", ""),
                        "protein_b": row.get("preferredName_B", ""),
                        "score": float(row.get("score", 0)),
                        "neighborhood": float(row.get("nscore", 0)),
                        "fusion": float(row.get("fscore", 0)),
                        "cooccurrence": float(row.get("pscore", 0)),
                        "coexpression": float(row.get("ascore", 0)),
                        "experimental": float(row.get("escore", 0)),
                        "database": float(row.get("dscore", 0)),
                        "textmining": float(row.get("tscore", 0)),
                    })

            return True

        except Exception as e:
            self.error = f"STRING 网络查询异常: {str(e)}"
            return False

    def get_results(self) -> list:
        """返回互作关系列表"""
        return self.interactions

    def get_summary(self) -> dict:
        """返回摘要统计"""
        if not self.interactions:
            return {}
        scores = [i["score"] for i in self.interactions]
        return {
            "query": self.identifier,
            "string_id": self.string_id,
            "preferred_name": self.preferred_name,
            "interaction_count": len(self.interactions),
            "max_score": max(scores),
            "mean_score": sum(scores) / len(scores),
        }

    def cleanup(self):
        self.interactions = []
        gc.collect()
