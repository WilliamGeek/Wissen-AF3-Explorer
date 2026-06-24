import subprocess
import os
import shutil
import sys
import gc
import tempfile
import pandas as pd


class FpocketRunner:
    """fpocket 高精度几何口袋检测器

    自动探测多种安装来源的 fpocket 引擎:
      1) 系统 PATH 直接可用
      2) Conda 环境 (bioconda 通道)
      3) WSL (Windows Subsystem for Linux)
      4) Linux / macOS 默认安装
    """

    # ── 引擎探测 ──────────────────────────

    # Windows cmd.exe 找不到命令时的特征文本
    _WIN_NOT_FOUND_PATTERNS = [
        "not recognized", "cannot find", "not found",
        "找不到", "无法识别"
    ]

    @staticmethod
    def _find_fpocket_bin() -> str:
        """探测 fpocket 可执行文件路径，按优先级依次尝试"""
        # 1. 优先使用 shutil.which 检查系统 PATH（跨平台安全）
        from shutil import which
        path_found = which("fpocket")
        if path_found:
            return path_found

        # 2. 常见 Conda 环境绝对路径
        conda_candidates = [
            os.path.expanduser("~/miniconda3/envs/bio/bin/fpocket"),
            os.path.expanduser("~/anaconda3/envs/bio/bin/fpocket"),
            os.path.expanduser("~/miniconda3/bin/fpocket"),
            os.path.expanduser("~/anaconda3/bin/fpocket"),
            os.path.expanduser("~/micromamba/envs/bio/bin/fpocket"),
        ]
        for cand in conda_candidates:
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand

        # 3. Linux 包管理器路径
        for cand in ["/usr/local/bin/fpocket", "/usr/bin/fpocket", "/opt/fpocket/bin/fpocket"]:
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand

        # 4. Conda run 模式
        for conda_cmd in ["conda", "micromamba"]:
            if which(conda_cmd):
                try:
                    result = subprocess.run(
                        [conda_cmd, "run", "-n", "base", "fpocket", "-h"],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0 or (result.returncode == 1 and
                            not any(p in result.stderr.lower() for p in
                                    FpocketRunner._WIN_NOT_FOUND_PATTERNS)):
                        return f"{conda_cmd} run -n base fpocket"
                except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                    pass

        # 5. WSL (Windows)
        if os.name == "nt":
            try:
                result = subprocess.run(
                    ["wsl", "which", "fpocket"],
                    capture_output=True, text=True, timeout=10
                )
                wsl_path = result.stdout.strip()
                if wsl_path and result.returncode == 0:
                    return f"wsl {wsl_path}"
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass

        return ""

    # ── WSL 路径转换 ──────────────────────

    @staticmethod
    def _win_to_wsl_path(win_path):
        """将 Windows 路径转换为 WSL 路径 (e.g. D:\\foo -> /mnt/d/foo)"""
        if os.name != "nt":
            return win_path
        # 处理盘符: D:\xxx -> /mnt/d/xxx
        if len(win_path) >= 2 and win_path[1] == ':':
            drive = win_path[0].lower()
            rest = win_path[2:].replace('\\', '/')
            # 移除开头的多余斜杠，避免 /mnt/d//path
            rest = rest.lstrip('/')
            return f"/mnt/{drive}/{rest}"
        return win_path.replace('\\', '/').replace('//', '/')

    # ── 公共接口 ──────────────────────────

    def __init__(self, file_path):
        self.file_path = file_path
        self.output_dir = None
        self.error = None
        self.top_pockets = []
        self.pocket_pdbs = {}
        self.fpocket_bin = ""
        self._cif_converted_pdb = None  # 如果 CIF 被转换为 PDB 供 fpocket
        self._is_wsl = False  # 是否通过 WSL 运行
        self._wsl_tmp_file = None  # WSL /tmp/ 中的临时文件路径

    def run(self):
        try:
            # Step 0: 探测引擎
            self.fpocket_bin = self._find_fpocket_bin()
            if not self.fpocket_bin:
                self.error = (
                    "未检测到 fpocket 引擎。\n\n"
                    "**安装指引** (选择任一方式):\n"
                    "1. Conda: `conda install -c bioconda fpocket`\n"
                    "2. 源码编译: https://github.com/Discngine/fpocket\n"
                    "3. WSL (Windows): `wsl apt install fpocket`\n\n"
                    "替代方案: 使用「成药口袋探测」网格法 或 CASTp 在线桥接。"
                )
                return False

            self._is_wsl = self.fpocket_bin.startswith("wsl ")

            # Step 1: 转换 CIF → PDB (fpocket 最兼容 PDB 格式)
            input_path = self.file_path
            if self.file_path.endswith(('.cif', '.cif.gz')):
                try:
                    input_path = self._cif_to_temp_pdb(self.file_path)
                    self._cif_converted_pdb = input_path
                    print(f"[fpocket] CIF 已转换为临时 PDB: {input_path}", file=sys.stderr)
                except Exception as conv_err:
                    print(f"[fpocket] CIF→PDB 转换失败，尝试直接使用原文件: {conv_err}", file=sys.stderr)

            # Step 2: 执行 fpocket
            if self._is_wsl:
                # WSL 模式: 将文件复制到 /tmp/ 避免路径空格导致 fpocket 崩溃
                wsl_tmp = self._copy_to_wsl_tmp(input_path)
                run_path = wsl_tmp
                self._wsl_tmp_file = wsl_tmp  # 记录以便清理
            else:
                run_path = input_path
                self._wsl_tmp_file = None

            cmd = self.fpocket_bin.split() + ["-f", run_path]
            print(f"[fpocket] 执行命令: {cmd}", file=sys.stderr)
            try:
                result = subprocess.run(
                    cmd, check=True, capture_output=True, text=True, timeout=120
                )
                print(f"[fpocket] stdout: {result.stdout[:300]}", file=sys.stderr)
                if result.stderr:
                    print(f"[fpocket] stderr: {result.stderr[:300]}", file=sys.stderr)
            except subprocess.CalledProcessError as e:
                self.error = f"fpocket 执行失败 (返回码 {e.returncode}):\nstdout: {e.stdout[:500]}\nstderr: {e.stderr[:500]}"
                self._cleanup_temp()
                return False
            except subprocess.TimeoutExpired:
                self.error = "fpocket 执行超时（超过 120 秒），结构可能过大。"
                self._cleanup_temp()
                return False

            # Step 3: 推断输出目录
            if self._is_wsl:
                wsl_out_dir = self._find_output_dir(run_path)
                out_dir = self._wsl_to_win_path(wsl_out_dir)
            else:
                out_dir = self._find_output_dir(input_path)
            self.output_dir = out_dir

            if not os.path.exists(out_dir):
                self.error = f"fpocket 运行完成，但未找到输出目录: {out_dir}"
                self._cleanup_temp()
                return False

            # Step 4: 解析结果
            # 使用 run_path 的 basename（fpocket 基于此命名输出文件）
            parse_name = os.path.splitext(os.path.basename(run_path))[0]
            if parse_name.endswith('_out'):
                parse_name = parse_name[:-4]
            self._parse_info(parse_name)

            # Step 5: 清理
            self._cleanup()
            self._cleanup_temp()
            return True

        except Exception as e:
            self.error = f"未知错误: {str(e)}"
            self._cleanup_temp()
            return False

    @staticmethod
    def _wsl_to_win_path(wsl_path):
        """将 WSL 路径转换回 Windows 路径

        - /mnt/d/foo -> D:\\foo
        - /tmp/foo   -> \\\\wsl$\\Debian\\tmp\\foo  (WSL 内部路径)
        """
        if os.name != "nt":
            return wsl_path
        import re
        # /mnt/d/xxx -> D:\xxx
        m = re.match(r'^/mnt/([a-z])(/.*)', wsl_path)
        if m:
            drive = m.group(1).upper()
            rest = m.group(2).replace('/', '\\')
            return f"{drive}:{rest}"
        # /tmp/xxx 或其他非 /mnt/ 路径 -> \\wsl$\{distro}\...
        if wsl_path.startswith('/'):
            distro = FpocketRunner._detect_wsl_distro()
            win_rest = wsl_path.replace('/', '\\')
            return f"\\\\wsl$\\{distro}{win_rest}"
        return wsl_path.replace('/', '\\')

    @staticmethod
    def _detect_wsl_distro():
        """检测默认 WSL 发行版名称

        注意: wsl -l -v 在 Windows 上输出 UTF-16LE 编码，需要特殊处理
        """
        try:
            result = subprocess.run(
                ["wsl", "-l", "-v"], capture_output=True, timeout=10
            )
            # wsl -l -v 输出 UTF-16LE 编码，需要解码
            raw = result.stdout
            if isinstance(raw, bytes):
                try:
                    text = raw.decode('utf-16-le', errors='ignore')
                except Exception:
                    text = raw.decode('utf-8', errors='ignore')
            else:
                text = raw

            # 清理 null 字节和多余空白
            text = text.replace('\x00', '').strip()
            lines = text.splitlines()

            for line in lines[1:]:  # 跳过标题行
                parts = line.split()
                if not parts:
                    continue
                # 默认发行版行以 * 开头
                if parts[0] == '*':
                    return parts[1] if len(parts) > 1 else 'Debian'

            # 如果没有找到带 * 的，返回第一个有效发行版
            for line in lines[1:]:
                parts = line.split()
                if parts and parts[0] not in ('NAME', ''):
                    return parts[0]
        except Exception:
            pass
        return 'Debian'

    # ── 内部方法 ──────────────────────────

    def _cif_to_temp_pdb(self, cif_path):
        """将 CIF 转换为临时 PDB 文件供 fpocket 使用

        注意: 在 WSL 模式下，临时文件必须放在 WSL 可读写的位置。
        Windows Temp 目录 (AppData) 在 WSL 中可能无写入权限，
        因此优先放在 cif_path 同目录下。
        """
        from utils.biopandas_patch import safe_read_mmcif
        import tempfile

        pmmcif = safe_read_mmcif(cif_path)

        # 优先放在 cif 同目录下，确保 WSL 可访问
        cif_dir = os.path.dirname(os.path.abspath(cif_path))
        try:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".pdb", delete=False, mode="w",
                encoding="utf-8", dir=cif_dir
            )
        except OSError:
            # 回退到系统默认临时目录
            tmp = tempfile.NamedTemporaryFile(
                suffix=".pdb", delete=False, mode="w", encoding="utf-8"
            )
        try:
            atom_df = pmmcif.df.get("ATOM")
            if atom_df is None or atom_df.empty:
                raise RuntimeError("CIF 文件中无 ATOM 数据块")

            tmp_path = tmp.name
            serial = 1  # PDB 序列号计数器
            for _, row in atom_df.iterrows():
                raw_id = row.get('id')
                serial = int(raw_id) if raw_id is not None and not (isinstance(raw_id, float) and pd.isna(raw_id)) else serial
                atom_name = row.get('label_atom_id') or row.get('atom_id') or 'C'
                atom_name = str(atom_name)
                if len(atom_name) < 4:
                    atom_name = f" {atom_name}" if len(atom_name) == 2 else f"  {atom_name}"
                alt_loc = row.get('label_alt_id') or ' '
                alt_loc = str(alt_loc) if alt_loc and alt_loc not in ('.', '?') else ' '
                res_name = row.get('label_comp_id') or row.get('auth_comp_id') or 'UNK'
                res_name = str(res_name)[:3]
                chain = row.get('label_asym_id') or row.get('auth_asym_id') or 'A'
                chain = str(chain)[:1]
                raw_seq = row.get('label_seq_id')
                if raw_seq is None or (isinstance(raw_seq, float) and pd.isna(raw_seq)):
                    raw_seq = row.get('auth_seq_id')
                res_seq = int(raw_seq) if raw_seq is not None and not (isinstance(raw_seq, float) and pd.isna(raw_seq)) else 1
                icode = row.get('pdbx_PDB_ins_code')
                if icode is None or (isinstance(icode, float) and pd.isna(icode)):
                    icode = ' '
                icode = str(icode)
                if icode in ('None', 'nan', '.', '?', ''):
                    icode = ' '
                icode = icode[:1]  # PDB 格式只允许 1 字符
                x = float(row.get('Cartn_x') or 0)
                y = float(row.get('Cartn_y') or 0)
                z = float(row.get('Cartn_z') or 0)
                occ = float(row.get('occupancy') or 1.0)
                bfac = float(row.get('B_iso_or_equiv') or 0.0)
                elem = str(row.get('type_symbol') or 'C')

                pdb_line = (
                    f"ATOM  {serial:5d} {atom_name:<4s}{alt_loc:1s}{res_name:3s} "
                    f"{chain:1s}{res_seq:4d}{icode:1s}   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{occ:6.2f}{bfac:6.2f}          {elem:>2s}\n"
                )
                tmp.write(pdb_line)
            tmp.write("END\n")
            tmp.close()
            del pmmcif
            gc.collect()
            return tmp_path
        except Exception:
            tmp.close()
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
            raise

    def _find_output_dir(self, input_path):
        """推断 fpocket 输出目录"""
        # 处理 WSL 路径 (forward slashes) 和 Windows 路径
        import re
        # 提取文件名
        if '/' in input_path:
            base = input_path.rsplit('/', 1)[-1]
            parent = input_path.rsplit('/', 1)[0] if '/' in input_path else ''
        else:
            base = os.path.basename(input_path)
            parent = os.path.dirname(input_path) or os.getcwd()

        for suffix in ['.pdb', '.cif', '.cif.gz']:
            if base.endswith(suffix):
                name = base[:-len(suffix)]
                break
        else:
            name = os.path.splitext(base)[0]

        if parent:
            return f"{parent}/{name}_out"
        return f"{os.getcwd()}/{name}_out"

    def _parse_info(self, name):
        info_file = os.path.join(self.output_dir, f"{name}_info.txt")
        if not os.path.exists(info_file):
            return

        pockets = []
        current_pocket = {}

        with open(info_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Pocket "):
                    if current_pocket:
                        pockets.append(current_pocket)
                    # "Pocket 45 :" → 提取数字 45
                    parts = line.split()
                    pocket_id = parts[1] if len(parts) >= 2 else ''
                    current_pocket = {'Pocket': pocket_id}
                elif ":" in line and current_pocket:
                    parts = line.split(":", 1)
                    key = parts[0].strip()
                    val = parts[1].strip()
                    try:
                        if key == "Score":
                            current_pocket['Pocket Score'] = float(val)
                        elif key == "Druggability Score":
                            current_pocket['Druggability Score'] = float(val)
                        elif key == "Volume":
                            current_pocket['Volume'] = float(val)
                        elif key == "Number of Alpha Spheres":
                            current_pocket['Alpha Spheres'] = int(val)
                        elif key == "Total SASA":
                            current_pocket['Total SASA'] = float(val)
                        elif key == "Hydrophobicity score":
                            current_pocket['Hydrophobicity'] = float(val)
                    except ValueError:
                        pass
            if current_pocket:
                pockets.append(current_pocket)

        # 按 Druggability Score 降序，取前 10 个
        pockets.sort(key=lambda x: x.get('Druggability Score', 0), reverse=True)
        self.top_pockets = pockets[:10]

        # 加载口袋 PDB 文件
        pockets_dir = os.path.join(self.output_dir, "pockets")
        if os.path.isdir(pockets_dir):
            for p in self.top_pockets:
                pocket_id = p.get('Pocket', '')
                if not pocket_id or pocket_id == ':':
                    continue
                pdb_file = os.path.join(pockets_dir, f"pocket{pocket_id}_atm.pdb")
                if os.path.exists(pdb_file):
                    with open(pdb_file, 'r', encoding='utf-8') as pf:
                        self.pocket_pdbs[f"pocket{pocket_id}"] = pf.read()

    def _copy_to_wsl_tmp(self, src_path):
        """将文件复制到 WSL /tmp/ 目录，避免路径空格导致 fpocket 崩溃

        返回 WSL 路径 (如 /tmp/xxx.pdb)
        """
        import uuid
        basename = os.path.basename(src_path)
        # 生成唯一文件名避免冲突
        unique_name = f"{uuid.uuid4().hex[:8]}_{basename}"
        wsl_tmp_path = f"/tmp/{unique_name}"

        # 使用 wsl cp 命令复制
        src_wsl = self._win_to_wsl_path(os.path.abspath(src_path))
        try:
            result = subprocess.run(
                ["wsl", "cp", src_wsl, wsl_tmp_path],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                raise RuntimeError(f"复制到 WSL /tmp/ 失败: {result.stderr}")
            print(f"[fpocket] 已复制到 WSL /tmp/: {wsl_tmp_path}", file=sys.stderr)
            return wsl_tmp_path
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"WSL 文件复制失败: {e}")

    def _cleanup(self):
        if self.output_dir and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir, ignore_errors=True)
        gc.collect()

    def _cleanup_temp(self):
        # 清理 WSL /tmp/ 中的临时文件
        if self._wsl_tmp_file:
            try:
                subprocess.run(
                    ["wsl", "rm", "-f", self._wsl_tmp_file],
                    capture_output=True, timeout=10
                )
            except Exception:
                pass
            self._wsl_tmp_file = None

        if self._cif_converted_pdb and os.path.exists(self._cif_converted_pdb):
            try:
                os.unlink(self._cif_converted_pdb)
            except OSError:
                pass
        gc.collect()

    def get_results_df(self):
        if not self.top_pockets:
            return pd.DataFrame()
        return pd.DataFrame(self.top_pockets)