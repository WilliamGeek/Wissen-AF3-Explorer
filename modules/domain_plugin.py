import requests
import gc
import sys
import time
import pandas as pd
from utils.biopandas_patch import safe_read_mmcif


class DomainAnnotator:
    """Pfam 蛋白结构域注释器 — 通过 EBI HMMER API 进行 Pfam 结构域快速注释
    
    API 参考: https://www.ebi.ac.uk/Tools/hmmer/
    支持同步和异步两种调用模式，自动适配不同版本 API 返回格式。
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.sequence = ""
        self.domains = []
        self.error = None
        self.seq_length = 0
        self._debug_info = []  # 调试信息收集
        self._extract_sequence()

    def _log(self, msg: str):
        """记录调试信息"""
        self._debug_info.append(msg)
        print(f"[Pfam] {msg}", file=sys.stderr)

    def _extract_sequence(self):
        try:
            pmmcif = safe_read_mmcif(self.file_path)

            # 方式1：从 entity_poly 读取序列
            try:
                seq_df = pmmcif.df.get('entity_poly')
                if seq_df is not None and not seq_df.empty and 'pdbx_seq_one_letter_code' in seq_df.columns:
                    raw_seq = seq_df['pdbx_seq_one_letter_code'].iloc[0]
                    self.sequence = raw_seq.replace('\n', '').replace(' ', '')
                    self._log(f"从 entity_poly 提取序列: {len(self.sequence)} aa")
            except KeyError:
                pass

            # 方式2：从 CA 原子推导序列
            if not self.sequence:
                try:
                    atom_df = pmmcif.df['ATOM']
                    atom_col = 'label_atom_id' if 'label_atom_id' in atom_df.columns else 'atom_name'
                    comp_col = 'label_comp_id' if 'label_comp_id' in atom_df.columns else 'residue_name'

                    ca_df = atom_df[atom_df[atom_col] == 'CA']
                    aa_map = {
                        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
                    }
                    seq_list = [aa_map.get(str(res)[:3], 'X') for res in ca_df[comp_col]]
                    self.sequence = "".join(seq_list).replace('X', '')
                    self._log(f"从 CA 原子推导序列: {len(self.sequence)} aa")
                except Exception as ex:
                    self._log(f"推导序列异常: {ex}")

                if 'ca_df' in locals():
                    del ca_df
            del pmmcif
            gc.collect()

            if not self.sequence:
                self.error = "未能成功从 CIF 文件中提取有效序列。"
                return

            self.seq_length = len(self.sequence)
        except Exception as e:
            self.error = f"解析结构提取序列时发生错误: {str(e)}"
            gc.collect()

    def run_annotation(self, progress_callback=None):
        """执行 Pfam 结构域注释

        Args:
            progress_callback: 可选回调函数 (status: str, progress: float) -> None
                用于向 UI 报告进度，progress 范围 0.0~1.0
        """
        if not self.sequence:
            return False

        self._log(f"序列长度: {self.seq_length} aa")
        self._log(f"序列前50: {self.sequence[:50]}...")
        if progress_callback:
            progress_callback("正在提交 EBI HMMER API...", 0.05)

        # ── 尝试多种 API 格式 ──
        formats = [
            # 格式 1: JSON body with 'input' field (EBI官方推荐)
            {
                "url": "https://www.ebi.ac.uk/Tools/hmmer/api/v1/search/hmmscan",
                "method": "json",
                "payload": {"database": "pfam", "input": f">query\n{self.sequence}"}
            },
        ]

        for attempt_idx, fmt in enumerate(formats):
            self._log(f"尝试格式 {attempt_idx + 1}: {fmt['method']}")
            if progress_callback:
                progress_callback("正在提交序列到 EBI 服务器...", 0.1)
            try:
                if fmt["method"] == "json":
                    headers = {"Content-Type": "application/json", "Accept": "application/json"}
                    response = requests.post(
                        fmt["url"], json=fmt["payload"], headers=headers, timeout=30
                    )
                else:
                    response = requests.post(
                        fmt["url"], files=fmt.get("files", {}),
                        data=fmt.get("data", {}), timeout=30
                    )

                self._log(f"HTTP {response.status_code}")

                if response.status_code != 200:
                    self._log(f"非 200 响应: {response.text[:200]}")
                    if progress_callback:
                        progress_callback(f"API 返回 {response.status_code}，提交失败", 0.0)
                    continue

                result = response.json()
                del response

                # ── 检查任务模式 ──
                job_id = result.get("id")
                has_direct_results = self._has_hits(result)

                if job_id and not has_direct_results:
                    # 异步模式：需要轮询
                    self._log(f"异步任务: {job_id}")
                    if progress_callback:
                        progress_callback("任务已提交，等待 EBI 服务器处理...", 0.15)
                    result = self._poll_result(job_id, progress_callback)
                    if result is None:
                        continue

                # 解析结果
                if progress_callback:
                    progress_callback("正在解析返回结果...", 0.85)
                self._parse_results(result)
                del result
                gc.collect()

                if self.domains:
                    self._log(f"成功！格式 {attempt_idx + 1} 找到 {len(self.domains)} 个结构域")
                    if progress_callback:
                        progress_callback(f"完成！检测到 {len(self.domains)} 个结构域", 1.0)
                    return True
                else:
                    self._log(f"格式 {attempt_idx + 1} 未找到结构域，尝试下一个...")
                    if progress_callback:
                        progress_callback("未检测到结构域", 0.0)

            except requests.exceptions.Timeout:
                self._log(f"格式 {attempt_idx + 1} 超时")
                if progress_callback:
                    progress_callback("API 请求超时，请检查网络连接", 0.0)
                continue
            except Exception as e:
                self._log(f"格式 {attempt_idx + 1} 异常: {e}")
                if progress_callback:
                    progress_callback(f"API 调用异常: {e}", 0.0)
                continue

        # 所有格式都失败
        if not self.domains:
            self.error = (
                "未检测到已知 Pfam 结构域。\n\n"
                "可能原因:\n"
                "1) 该蛋白不含已知的 Pfam-A 结构域\n"
                "2) EBI HMMER 服务器繁忙，请稍后重试\n"
                f"调试信息: {'; '.join(self._debug_info[-5:])}"
            )
        return False

    def _has_hits(self, result):
        """检查响应中是否直接包含 hits"""
        try:
            # API 返回 'result' (单数)，兼容 'results' (复数)
            results_data = result.get('result') or result.get('results')
            if isinstance(results_data, list):
                for item in results_data:
                    if isinstance(item, dict) and item.get('hits'):
                        return True
            elif isinstance(results_data, dict) and results_data.get('hits'):
                return True
            if isinstance(result, dict) and result.get('hits'):
                return True
        except Exception:
            pass
        return False

    def _poll_result(self, job_id, progress_callback=None):
        """轮询异步任务结果

        EBI HMMER API 轮询策略:
        - 直接请求 /result/{job_id} 端点
        - 响应中包含 status 字段，可能的值: SUCCESS, PENDING, RUNNING, ERROR
        - 最多轮询 60 次 × 3s = 180 秒，超时则放弃
        """
        result_url = f"https://www.ebi.ac.uk/Tools/hmmer/api/v1/result/{job_id}"
        max_retries = 60  # 最多等 3 分钟 (60 × 3s)，长序列 hmmscan 需要更多时间
        poll_interval = 3  # 每 3 秒轮询一次

        for attempt in range(max_retries):
            time.sleep(poll_interval)
            try:
                resp = requests.get(result_url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                job_status = str(data.get("status", "")).upper()

                # 计算进度 (0.15 ~ 0.80)
                progress = 0.15 + (0.65 * (attempt + 1) / max_retries)

                # 每 5 次报告一次状态
                if attempt % 5 == 0:
                    status_text = f"等待 EBI 处理... ({attempt + 1}/{max_retries}) [{job_status}]"
                    self._log(f"轮询 {attempt + 1}: status={job_status}")
                    if progress_callback:
                        progress_callback(status_text, progress)

                if job_status == "SUCCESS":
                    self._log(f"任务完成 (attempt {attempt + 1})")
                    if progress_callback:
                        progress_callback("EBI 处理完成，正在解析...", 0.85)
                    return data

                if job_status in ("ERROR", "FAILED"):
                    self._log(f"任务失败: {job_status}")
                    if progress_callback:
                        progress_callback(f"EBI 任务失败: {job_status}", 0.0)
                    return None

                # PENDING / RUNNING 状态继续等待

            except Exception as e:
                self._log(f"轮询异常: {e}")
                continue

        self._log("轮询超时 (180秒)")
        if progress_callback:
            progress_callback("EBI 服务器响应超时 (180秒)，请稍后重试", 0.0)
        return None

    def _parse_results(self, res_json):
        """多层次解析 HMMER API 结果"""
        try:
            # Layer 1: 提取结果容器 (API 返回 'result' 单数)
            results_data = res_json.get('result') or res_json.get('results')
            if results_data is None:
                results_data = res_json

            # Layer 2: 如果是列表，取第一个元素
            if isinstance(results_data, list):
                if len(results_data) > 0:
                    results_data = results_data[0]
                else:
                    return

            if not isinstance(results_data, dict):
                return

            # Layer 3: 提取 hits
            hits = results_data.get('hits', [])
            self._log(f"解析到 {len(hits)} 个 hits")

            for hit in hits:
                # 名称优先取 metadata.identifier，回退到 name
                metadata = hit.get('metadata', {})
                name = metadata.get('identifier') or hit.get('name', 'Unknown')
                desc = metadata.get('description') or hit.get('desc', hit.get('description', 'No description'))
                accession = metadata.get('accession') or hit.get('acc', hit.get('accession', ''))

                # hmmscan 可能包含多个结构域命中 (domains 或 doms)
                domains = hit.get('domains', hit.get('doms', []))
                if not domains:
                    # 单个 domain = hit 本身
                    sqfrom = hit.get('sqfrom', hit.get('ali_from', hit.get('env_from', hit.get('iali', 0))))
                    sqto = hit.get('sqto', hit.get('ali_to', hit.get('env_to', hit.get('jali', 0))))
                    ievalue = hit.get('ievalue', hit.get('evalue', 'N/A'))
                    self.domains.append({
                        'Domain': name,
                        'Description': desc,
                        'Accession': accession,
                        'Start': int(sqfrom) if sqfrom else 0,
                        'End': int(sqto) if sqto else 0,
                        'E-value': str(ievalue),
                    })
                else:
                    for dom in domains:
                        sqfrom = dom.get('sqfrom', dom.get('ali_from', dom.get('env_from', dom.get('iali', 0))))
                        sqto = dom.get('sqto', dom.get('ali_to', dom.get('env_to', dom.get('jali', 0))))
                        ievalue = dom.get('ievalue', dom.get('i_evalue', dom.get('independent_evalue', 'N/A')))
                        self.domains.append({
                            'Domain': name,
                            'Description': desc,
                            'Accession': accession,
                            'Start': int(sqfrom) if sqfrom else 0,
                            'End': int(sqto) if sqto else 0,
                            'E-value': str(ievalue),
                        })
        except Exception as e:
            self._log(f"解析结果异常: {e}")

    def get_results_df(self):
        if not self.domains:
            return pd.DataFrame(columns=['Domain', 'Description', 'Accession', 'Start', 'End'])
        return pd.DataFrame(self.domains)

    def get_debug_info(self):
        """返回调试信息列表"""
        return self._debug_info