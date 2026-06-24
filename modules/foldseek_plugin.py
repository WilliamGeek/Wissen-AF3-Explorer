import requests
import time
import gc
import sys
import os
import tempfile
import pandas as pd
import gzip


class FoldseekAPIWrapper:
    """Foldseek 结构相似性远程检索封装器
    
    自动将 CIF 转换为 PDB 后再上传，确保与 EBI Foldseek 服务器的格式兼容。
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.error = None
        self.top_results = []
        self.base_url = "https://search.foldseek.com/api"
        self._temp_pdb = None  # CIF→PDB 临时文件

    def run(self):
        try:
            # Step 1: 读取结构内容 — CIF 自动转为 PDB
            file_content, file_ext = self._read_file(self.file_path)
            if not file_content:
                self.error = "无法读取结构文件内容"
                return False

            # Step 2: 提交任务到 Foldseek 服务器 (带重试)
            filename = f"query.{file_ext}"
            files = {'q': (filename, file_content, 'application/octet-stream')}
            data = [
                ('mode', '3diaa'),
                ('database[]', 'pdb100'),
            ]

            ticket_id = None
            for attempt in range(3):
                try:
                    response = requests.post(
                        f"{self.base_url}/ticket",
                        files=files,
                        data=data,
                        timeout=60
                    )
                    if response.status_code in (502, 503, 504):
                        print(f"[Foldseek] 提交任务失败 ({response.status_code})，重试 {attempt+1}/3", file=sys.stderr)
                        time.sleep(3 * (attempt + 1))
                        continue
                    response.raise_for_status()
                    ticket_data = response.json()
                    ticket_id = ticket_data.get('id')
                    del response
                    break
                except requests.exceptions.HTTPError as e:
                    if attempt < 2:
                        print(f"[Foldseek] 提交任务 HTTP 错误: {e}，重试 {attempt+1}/3", file=sys.stderr)
                        time.sleep(3 * (attempt + 1))
                    else:
                        raise

            if not ticket_id:
                self.error = "未能获取 Foldseek 任务 ticket ID（服务器可能暂时不可用，请稍后重试）"
                self._cleanup_temp()
                return False

            del file_content
            gc.collect()

            # Step 3: 轮询任务状态
            status = "RUNNING"
            max_retries = 30
            retries = 0
            while status in ["RUNNING", "PENDING"] and retries < max_retries:
                time.sleep(2)
                try:
                    resp_status = requests.get(
                        f"{self.base_url}/ticket/{ticket_id}",
                        timeout=15
                    )
                    if resp_status.status_code in (502, 503, 504):
                        print(f"[Foldseek] 轮询状态 {resp_status.status_code}，等待重试", file=sys.stderr)
                        retries += 1
                        continue
                    resp_status.raise_for_status()
                    status_data = resp_status.json()
                    status = status_data.get("status", "UNKNOWN")
                    del resp_status
                except Exception as poll_err:
                    print(f"[Foldseek] 轮询异常: {poll_err}", file=sys.stderr)
                    retries += 1
                    continue
                retries += 1

            if status != "COMPLETE":
                self.error = f"Foldseek 任务未完成 (状态: {status})，请稍后重试"
                self._cleanup_temp()
                return False

            # Step 4: 获取结果 (带重试)
            res_json = None
            resp_res = None
            for attempt in range(3):
                try:
                    resp_res = requests.get(
                        f"{self.base_url}/result/{ticket_id}/0",
                        timeout=30
                    )
                    if resp_res.status_code in (502, 503, 504):
                        print(f"[Foldseek] 获取结果 {resp_res.status_code}，重试 {attempt+1}/3", file=sys.stderr)
                        time.sleep(3 * (attempt + 1))
                        resp_res = None
                        continue
                    resp_res.raise_for_status()
                    res_json = resp_res.json()
                    break
                except requests.exceptions.HTTPError as e:
                    resp_res = None
                    if attempt < 2:
                        print(f"[Foldseek] 获取结果 HTTP 错误: {e}，重试 {attempt+1}/3", file=sys.stderr)
                        time.sleep(3 * (attempt + 1))
                    else:
                        raise

            if res_json is None:
                self.error = "Foldseek 结果获取失败（服务器可能暂时不可用，请稍后重试）"
                self._cleanup_temp()
                return False

            # 调试：打印返回结构摘要
            print(f"[Foldseek] API 响应类型: {type(res_json).__name__}", file=sys.stderr)
            if isinstance(res_json, dict):
                print(f"[Foldseek] 响应 keys: {list(res_json.keys())}", file=sys.stderr)
                for key, val in res_json.items():
                    if isinstance(val, (list, dict)):
                        print(f"[Foldseek]   {key}: {type(val).__name__} len={len(val)}", file=sys.stderr)
                    else:
                        print(f"[Foldseek]   {key}: {val}", file=sys.stderr)
            elif isinstance(res_json, list):
                print(f"[Foldseek] 返回列表，长度={len(res_json)}", file=sys.stderr)
                if len(res_json) > 0 and isinstance(res_json[0], dict):
                    print(f"[Foldseek] 首个元素 keys: {list(res_json[0].keys())}", file=sys.stderr)
                    alns = res_json[0].get('alignments', [])
                    print(f"[Foldseek] 首个元素 alignments 数量: {len(alns)}", file=sys.stderr)
            else:
                print(f"[Foldseek] 响应: {str(res_json)[:500]}", file=sys.stderr)

            self._parse_results(res_json)

            del resp_res
            del res_json
            gc.collect()
            self._cleanup_temp()

            return True

        except requests.exceptions.Timeout:
            self.error = "Foldseek API 请求超时（网络环境可能不稳定，建议使用代理或稍后重试）"
            self._cleanup_temp()
            return False
        except requests.exceptions.ConnectionError:
            self.error = "无法连接到 Foldseek 服务器，请检查网络链接或 VPN 状态"
            self._cleanup_temp()
            return False
        except Exception as e:
            self.error = f"Foldseek API 交互异常: {str(e)}"
            self._cleanup_temp()
            return False

    def _read_file(self, file_path):
        """读取结构文件，CIF 格式自动转换为 PDB"""
        try:
            content = None
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    content = f.read()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

            if not content:
                return None, "pdb"

            # CIF → PDB 自动转换
            if file_path.endswith(('.cif', '.cif.gz')):
                try:
                    pdb_content = self._cif_to_pdb_string(file_path)
                    return pdb_content, "pdb"
                except Exception as conv_err:
                    print(f"[Foldseek] CIF→PDB 转换失败，尝试直接发送 CIF: {conv_err}", file=sys.stderr)
                    return content, "cif"

            return content, "pdb"
        except Exception as e:
            print(f"[Foldseek] 读取文件失败: {e}", file=sys.stderr)
            return None, "pdb"

    def _cif_to_pdb_string(self, cif_path):
        """CIF → PDB 格式转换 (严格 PDB 列对齐)"""
        from utils.biopandas_patch import safe_read_mmcif

        pmmcif = safe_read_mmcif(cif_path)
        atom_df = pmmcif.df.get("ATOM")
        if atom_df is None or atom_df.empty:
            raise RuntimeError("CIF 文件中无 ATOM 数据块")

        lines = []
        serial = 1
        for _i, row in atom_df.iterrows():
            # 跳过氢原子 (Foldseek 通常只关心重原子)
            elem = str(row.get('type_symbol', 'C'))[:2]
            if elem == 'H':
                continue

            atom_name = str(row.get('label_atom_id', row.get('atom_id', ' CA')))
            # PDB 标准: 4-char atom name, right-justified with leading space for 1-3 char names
            atom_name = atom_name.strip()
            if len(atom_name) <= 3:
                atom_name = f" {atom_name:<3s}"  # e.g. " CA ", " N  ", " O  "
            else:
                atom_name = f"{atom_name:<4s}"  # e.g. "OXT " (4-char names start at col 13)

            alt_loc = str(row.get('label_alt_id', ' '))
            if alt_loc in ('.', '', '?', 'None', 'nan'):
                alt_loc = ' '
            alt_loc = alt_loc[:1]  # PDB 格式只允许 1 字符
            res_name = str(row.get('label_comp_id', row.get('auth_comp_id', 'UNK')))[:3]
            # 防止 res_name 为 "Non" (来自 "None")
            if res_name in ('Non', 'nan', 'non', 'NON'):
                res_name = 'UNK'
            chain = str(row.get('label_asym_id', row.get('auth_asym_id', 'A')))
            if chain in ('None', 'nan', '', '.', '?'):
                chain = 'A'
            chain = chain[:1]
            res_seq = int(row.get('label_seq_id', row.get('auth_seq_id', 1)))
            icode = str(row.get('pdbx_PDB_ins_code', ' '))
            if icode in ('?', '.', '', 'None', 'nan') or icode is None:
                icode = ' '
            icode = icode[:1]  # PDB 格式只允许 1 字符
            x = float(row.get('Cartn_x', 0))
            y = float(row.get('Cartn_y', 0))
            z = float(row.get('Cartn_z', 0))
            occ = float(row.get('occupancy', 1.0))
            bfac = float(row.get('B_iso_or_equiv', 0.0))

            # 标准 PDB ATOM 格式: columns 1-80
            # 1-6:"ATOM  " 7-11:serial 13-16:atomName 17:altLoc 18-20:resName
            # 22:chainID 23-26:resSeq 27:iCode 31-38:x 39-46:y 47-54:z
            # 55-60:occupancy 61-66:tempFactor 77-78:element
            lines.append(
                f"ATOM  {serial:5d} {atom_name}{alt_loc}{res_name:>3s} "
                f"{chain}{res_seq:4d}{icode}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{occ:6.2f}{bfac:6.2f}          {elem:>2s}"
            )
            serial += 1
        lines.append("END")
        del pmmcif
        gc.collect()
        return "\n".join(lines)

    def _parse_results(self, res_json):
        """解析 Foldseek 返回的 JSON 结果，提取前 10 个最佳匹配"""
        results = []
        try:
            alignments = []

            # Foldseek API 返回格式: {"results": [{"alignments": [...]}]}
            if isinstance(res_json, dict):
                if 'alignments' in res_json:
                    alignments = res_json['alignments']
                elif 'results' in res_json:
                    # 新版 API: results 是列表，每个元素包含 alignments
                    results_list = res_json['results']
                    print(f"[Foldseek] results_list type: {type(results_list)}, len: {len(results_list) if isinstance(results_list, (list, dict)) else 'N/A'}", file=sys.stderr)
                    if isinstance(results_list, list):
                        for idx, item in enumerate(results_list):
                            print(f"[Foldseek] results[{idx}] type: {type(item)}, keys: {list(item.keys()) if isinstance(item, dict) else 'N/A'}", file=sys.stderr)
                            if isinstance(item, dict) and 'alignments' in item:
                                alns = item['alignments']
                                print(f"[Foldseek]   alignments len: {len(alns) if isinstance(alns, list) else 'N/A'}", file=sys.stderr)
                                if isinstance(alns, list) and len(alns) > 0:
                                    print(f"[Foldseek]   first alignment type: {type(alns[0]).__name__}", file=sys.stderr)
                                    if isinstance(alns[0], dict):
                                        print(f"[Foldseek]   first alignment keys: {list(alns[0].keys())}", file=sys.stderr)
                                    elif isinstance(alns[0], list):
                                        print(f"[Foldseek]   first alignment is list, len={len(alns[0])}", file=sys.stderr)
                                        if len(alns[0]) > 0:
                                            print(f"[Foldseek]     sub-item type: {type(alns[0][0]).__name__}", file=sys.stderr)
                                            if isinstance(alns[0][0], dict):
                                                print(f"[Foldseek]     sub-item keys: {list(alns[0][0].keys())}", file=sys.stderr)
                                    print(f"[Foldseek]   raw first alignment: {str(alns[0])[:300]}", file=sys.stderr)

                                # 处理嵌套列表: alignments = [[dict, dict, ...], ...]
                                for aln_group in alns:
                                    if isinstance(aln_group, list):
                                        # 嵌套列表，展开子项
                                        alignments.extend(aln_group)
                                    elif isinstance(aln_group, dict):
                                        alignments.append(aln_group)
                    elif isinstance(results_list, dict) and 'alignments' in results_list:
                        alignments = results_list['alignments']
            elif isinstance(res_json, list) and len(res_json) > 0:
                for item in res_json:
                    if isinstance(item, dict) and 'alignments' in item:
                        alignments.extend(item['alignments'])

            if not alignments:
                return

            for aln in alignments:
                target = aln.get('target', '')
                if not target:
                    continue
                pdb_id = target.split('_')[0] if '_' in target else target
                pdb_id = pdb_id.split('-')[0]

                # Foldseek 3D 模式: prob (0-100) 比 score 更接近 TM-score
                prob = aln.get('prob', 0)
                raw_score = aln.get('score', 0)
                # prob 范围 0-100, TM-score 范围 0-1
                tm_score = float(prob) / 100.0 if prob else 0.0
                seq_id = aln.get('seqId', aln.get('seqIdentity', aln.get('seq_id', 0.0)))
                e_value = aln.get('eval', aln.get('evalue', aln.get('eValue', 'N/A')))
                aln_len = aln.get('alnLength', aln.get('alnlen', aln.get('length', 0)))

                results.append({
                    'PDB ID': pdb_id,
                    'TM-score': round(tm_score, 4),
                    'Probability': round(float(prob), 2) if prob else 0,
                    'Sequence Identity': float(seq_id),
                    'E-value': str(e_value),
                    'Alignment Length': int(aln_len) if aln_len else 0,
                    'Target': target
                })
        except Exception as parse_err:
            print(f"[Foldseek] 结果解析异常: {parse_err}", file=sys.stderr)

        if results:
            results.sort(key=lambda x: x.get('TM-score', 0), reverse=True)
            self.top_results = results[:10]
        else:
            self.error = (
                "Foldseek 检索完成但未找到匹配结构。可能原因:\n"
                "1) 蛋白结构非常新颖，PDB 中无同源结构\n"
                "2) CIF 文件格式不兼容 — 已自动转换为 PDB 后重试\n\n"
                "建议: 尝试 CASTp 或 PAE 分析作为替代方案。"
            )

    def _cleanup_temp(self):
        if self._temp_pdb and os.path.exists(self._temp_pdb):
            try:
                os.unlink(self._temp_pdb)
            except OSError:
                pass
        gc.collect()

    def get_results_df(self):
        if not self.top_results:
            return pd.DataFrame(columns=['PDB ID', 'TM-score', 'Sequence Identity', 'E-value', 'Alignment Length'])
        return pd.DataFrame(self.top_results)