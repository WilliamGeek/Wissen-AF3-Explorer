"""
Wissen AF3 Explorer v3.1 — AlphaFold 3 输出文件二次分析一站式平台
主入口：Streamlit 交互式 Web 前端
架构: 三视图导航 (介绍 → 仪表盘 → 工具) + 跨工具缓存 + 现代 UI
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import gc
import time
from pathlib import Path
from stmol import showmol
import py3Dmol
import plotly.express as px

from core.zip_parser import AF3ZipParser, AF3ValidationError
from core.viewer_3d import render_protein, render_protein_with_pocket
from core.ai_reporter import generate_insight_report
from core.exporter import MarkdownExporter
from core.pocket_detector import PocketDetector
from core.cache_manager import CacheManager
from utils.biopandas_patch import safe_read_mmcif


# ════════════════════════════════════════════════
# 工具函数: CASTp POC 文件解析
# ════════════════════════════════════════════════

def _parse_castp_poc(content):
    """解析 CASTp .poc 文件，返回口袋摘要列表

    CASTp POC 文件是 PDB ATOM 格式:
    ATOM  37  CB  TRP A   6     -16.693  83.733  10.530  1.00 36.21 66 POC
    字段: ATOM serial name resName chainID resSeq x y z occupancy bFactor pocketID POC

    按 pocketID 分组统计每个口袋的体积(原子数)和残基组成。
    """
    from collections import defaultdict

    pocket_atoms = defaultdict(list)  # pocket_id -> [atom_records]

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('HEADER'):
            continue

        # 只处理 ATOM/HETATM 行且以 POC 结尾
        if not (stripped.startswith('ATOM') or stripped.startswith('HETATM')):
            continue
        if not stripped.endswith('POC'):
            continue

        # PDB 固定列格式解析
        try:
            # ATOM  1-6, serial 7-11, name 13-16, resName 18-20, chainID 22,
            # resSeq 23-26, x 31-38, y 39-46, z 47-54, occupancy 55-60, bFactor 61-66, pocketID 67-70
            if len(stripped) < 54:
                continue
            atom_name = stripped[12:16].strip()
            res_name = stripped[17:20].strip()
            chain_id = stripped[21:22].strip()
            res_seq = stripped[22:26].strip()
            x = float(stripped[30:38])
            y = float(stripped[38:46])
            z = float(stripped[46:54])
            # 口袋 ID 在 bFactor 之后
            pocket_id = stripped[66:70].strip() if len(stripped) > 70 else ''
            if not pocket_id:
                # 回退: 取 POC 前面的数字
                parts = stripped.split()
                for p in reversed(parts[:-1]):
                    if p.replace('.', '').isdigit():
                        pocket_id = p
                        break

            if pocket_id:
                pocket_atoms[pocket_id].append({
                    'atom': atom_name,
                    'res': f"{chain_id}{res_seq}{res_name}",
                    'chain': chain_id,
                    'resSeq': res_seq,
                    'resName': res_name,
                    'x': x, 'y': y, 'z': z,
                })
        except (ValueError, IndexError):
            continue

    # 汇总每个口袋
    pockets = []
    for pid, atoms in sorted(pocket_atoms.items(), key=lambda x: -len(x[1])):
        residues = []
        seen = set()
        for a in atoms:
            key = a['res']
            if key not in seen:
                seen.add(key)
                residues.append(key)

        # 估算体积: 用原子包围盒的近似体积
        xs = [a['x'] for a in atoms]
        ys = [a['y'] for a in atoms]
        zs = [a['z'] for a in atoms]
        dx, dy, dz = max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)
        approx_volume = round(dx * dy * dz, 2) if dx and dy and dz else round(len(atoms) * 10, 2)

        pockets.append({
            'Pocket ID': pid,
            'Atoms': len(atoms),
            'Residues': len(residues),
            'Est. Volume (Å³)': approx_volume,
            'Center (x,y,z)': f"({round(sum(xs)/len(xs),1)}, {round(sum(ys)/len(ys),1)}, {round(sum(zs)/len(zs),1)})",
            'Residue List': '; '.join(residues[:12]) + (f' ... (+{len(residues)-12})' if len(residues) > 12 else ''),
        })

    return pockets


# ════════════════════════════════════════════════
# 1. 全局页面配置 & 现代 CSS
# ════════════════════════════════════════════════
st.set_page_config(
    layout="wide",
    page_title="Wissen AF3 Explorer",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── 全局变量 ── */
:root {
    --primary: #6366f1; --primary-light: #818cf8; --primary-dark: #4f46e5;
    --accent: #06b6d4; --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
    --bg: #0f172a; --bg-card: #1e293b; --bg-card-hover: #334155;
    --border: #334155; --text: #e2e8f0; --text-muted: #94a3b8;
}

/* ── 全局背景 ── */
.stApp { background-color: var(--bg); }
.main .block-container { padding-top: 1rem; }

/* ── 标题 ── */
h1 { color: var(--text) !important; font-weight: 700 !important; font-size: 1.8rem !important; }
h2 { color: var(--text) !important; font-weight: 600 !important; font-size: 1.4rem !important; }
h3 { color: var(--text) !important; font-weight: 600 !important; font-size: 1.1rem !important; }
p, li, span, div { color: var(--text-muted); }

/* ── 卡片 ── */
.tool-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.2rem; margin-bottom: 0.8rem;
    transition: all 0.2s ease; cursor: pointer;
}
.tool-card:hover { background: var(--bg-card-hover); border-color: var(--primary); transform: translateY(-2px); }
.tool-card .icon { font-size: 1.8rem; margin-bottom: 0.3rem; }
.tool-card .name { color: var(--text); font-weight: 600; font-size: 1rem; margin-bottom: 0.2rem; }
.tool-card .desc { color: var(--text-muted); font-size: 0.82rem; line-height: 1.4; }

/* ── 徽章 ── */
.badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-analyzed { background: rgba(16,185,129,0.15); color: var(--success); border: 1px solid rgba(16,185,129,0.3); }
.badge-unused { background: rgba(148,163,184,0.1); color: var(--text-muted); border: 1px solid rgba(148,163,184,0.2); }

/* ── 分类区块标题 ── */
.category-header {
    display: flex; align-items: center; gap: 0.5rem; margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 0.4rem; border-bottom: 2px solid var(--border);
}
.category-header .cat-icon { font-size: 1.3rem; }
.category-header .cat-name { color: var(--text); font-weight: 700; font-size: 1.05rem; }
.category-header .cat-desc { color: var(--text-muted); font-size: 0.8rem; margin-left: 0.5rem; }

/* ── 指标卡片 ── */
.metric-card {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-card-hover));
    border: 1px solid var(--border); border-radius: 10px; padding: 0.9rem 1.2rem;
}
.metric-card .label { color: var(--text-muted); font-size: 0.78rem; font-weight: 500; }
.metric-card .value { color: var(--text); font-size: 1.6rem; font-weight: 700; }

/* ── 分割线 ── */
hr { border-color: var(--border) !important; margin: 0.8rem 0 !important; }

/* ── 按钮微调 ── */
.stButton > button {
    border-radius: 8px !important; font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { transform: translateY(-1px); }

/* ── 侧边栏 ── */
[data-testid="stSidebar"] { background: var(--bg); border-right: 1px solid var(--border); }

/* ── Toast / 通知 ── */
[data-testid="stNotification"] { border-radius: 10px !important; }

/* ── Expander ── */
[data-testid="stExpander"] { border-radius: 10px !important; border-color: var(--border) !important; }

/* ── Tab 美化 ── */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0; padding: 0.5rem 1rem;
    background: var(--bg-card); color: var(--text-muted);
}
.stTabs [aria-selected="true"] { background: var(--primary) !important; color: #fff !important; }

/* ── 响应式 ── */
@media (max-width: 768px) { .tool-card { padding: 0.8rem; } }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
# 2. 会话状态初始化
# ════════════════════════════════════════════════
if "parser" not in st.session_state:
    st.session_state.parser = AF3ZipParser()
if "page_view" not in st.session_state:
    st.session_state.page_view = "intro"
if "selected_tool" not in st.session_state:
    st.session_state.selected_tool = None

CacheManager.init()

# ════════════════════════════════════════════════
# 3. 工具注册表加载
# ════════════════════════════════════════════════
@st.cache_data
def load_config():
    config_path = Path(__file__).parent / "configs" / "tools_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

cfg = load_config()
all_tools = {t["id"]: t for t in cfg["tools"] if t.get("enabled", True)}
categories = {c["id"]: c for c in cfg["categories"]}

def get_category_tools(cat_id: str):
    return [t for t in all_tools.values() if t.get("category") == cat_id]

def _tool_status(tool_id: str) -> str:
    return "analyzed" if CacheManager.is_analyzed(tool_id) else "unused"

# ════════════════════════════════════════════════
# 4. 侧边栏 — 导航枢纽
# ════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧬 Wissen AF3 Explorer")
    st.markdown("---")

    # 文件上传 (始终可见)
    uploaded_file = st.file_uploader(
        "📤 上传 AF3 输出压缩包",
        type=["zip"],
        help="Google DeepMind AlphaFold 3 Server 标准全量 .zip 压缩包"
    )

    st.markdown("---")

    # 导航按钮
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("🏠 仪表盘", use_container_width=True, key="nav_dashboard",
                     disabled=(st.session_state.page_view == "dashboard")):
            st.session_state.page_view = "dashboard"
            st.session_state.selected_tool = None
            st.rerun()
    with nav_col2:
        if st.button("📋 介绍页", use_container_width=True, key="nav_intro",
                     disabled=(st.session_state.page_view == "intro")):
            st.session_state.page_view = "intro"
            st.session_state.selected_tool = None
            st.rerun()

    st.markdown("---")

    # 快捷工具选择 (仅在仪表盘/工具视图)
    if st.session_state.page_view in ("dashboard", "tool") and uploaded_file is not None:
        st.caption("⚡ 快捷工具")
        tool_names = {tid: f"{t['icon']} {t['name']}" for tid, t in all_tools.items()}
        quick_select = st.selectbox(
            "跳转到工具", options=list(tool_names.keys()),
            format_func=lambda x: tool_names[x], key="quick_tool_sel", label_visibility="collapsed"
        )
        if st.button("▶ 启动", use_container_width=True, key="quick_launch"):
            st.session_state.selected_tool = quick_select
            st.session_state.page_view = "tool"
            st.rerun()

    st.markdown("---")

    # AI 设置
    with st.expander("⚙️ AI 洞察设置", expanded=False):
        api_base = st.text_input("API Base URL", value="http://localhost:3000/v1")
        api_key = st.text_input("API Key", type="password")
        model_name = st.text_input("模型名称", value="gemini-3.1-pro")

    st.markdown("---")

    # 缓存信息
    analyzed_count = CacheManager.get_analyzed_count()
    if analyzed_count > 0:
        st.caption(f"💾 缓存: {analyzed_count} 个工具已分析")

    # 重置
    if st.button("🧹 清空全部缓存并重置", use_container_width=True):
        st.session_state.parser.cleanup()
        keep_keys = {"parser"}
        for key in list(st.session_state.keys()):
            if key not in keep_keys:
                del st.session_state[key]
        CacheManager.reset()
        st.session_state.page_view = "intro"
        st.session_state.parser = AF3ZipParser()
        st.toast("✨ 所有缓存已清空，系统已完全重置！")
        st.rerun()

    st.markdown("---")
    st.caption("Wissen 解耦插件化架构 | © 2025")

# ════════════════════════════════════════════════
# 5. 主内容区视图路由
# ════════════════════════════════════════════════

# ── 5a. 介绍页 ──
if st.session_state.page_view == "intro" or uploaded_file is None:
    st.session_state.page_view = "intro"
    st.title("🧬 Wissen AF3 Explorer")

    # ── 分析流程 (全宽) ──
    st.markdown("### 📊 分析流程")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**1. 上传** ")
        st.caption("在侧边栏上传 AF3 标准 .zip 压缩包")
    with c2:
        st.markdown("**2. 解析** 🔍")
        st.caption("自动解压、扫描、提取关键数据")
    with c3:
        st.markdown("**3. 分析** ")
        st.caption("从仪表盘选择工具进行深度分析")
    with c4:
        st.markdown("**4. 报告** ")
        st.caption("AI 汇总全部分析结果生成报告")

    st.markdown("---")

    # ── 内置工具列表 ──
    st.markdown("### 🧰 内置工具")
    tool_cols = st.columns(3)
    for idx, cat in enumerate(cfg["categories"]):
        cat_tools = get_category_tools(cat["id"])
        if cat_tools:
            with tool_cols[idx % 3]:
                st.markdown(f"**{cat['icon']} {cat['name']}** ({len(cat_tools)} 个)")
                for t in cat_tools:
                    st.markdown(f"- {t['icon']} {t['name']}")

    st.markdown("---")
    st.info("💡 **开始使用**: 请在左侧侧边栏上传 AlphaFold 3 的标准输出 `.zip` 压缩包，系统将自动进入分析仪表盘。")

# ── 5b. 文件已上传 → 仪表盘或工具视图 ──
elif uploaded_file is not None:
    # 文件解析 (新文件才重新解析)
    try:
        if ("last_uploaded_name" not in st.session_state or
                st.session_state.last_uploaded_name != uploaded_file.name):
            st.session_state.parser.cleanup()
            # 清除旧缓存
            clear_keys = ["data_json", "ranking_df", "best_cif_path", "cif_content",
                          "context_data", "pocket_result"]
            for k in clear_keys:
                st.session_state.pop(k, None)
            CacheManager.reset()
            with st.spinner("🚀 后端解析引擎正在执行安全解压与深度扫描..."):
                data_json, ranking_df = st.session_state.parser.parse(uploaded_file)
                st.session_state.data_json = data_json
                st.session_state.ranking_df = ranking_df
                st.session_state.best_cif_path = st.session_state.parser.get_best_cif_path()
                with open(st.session_state.best_cif_path, "r", encoding="utf-8") as _f:
                    st.session_state.cif_content = _f.read()
                st.session_state.last_uploaded_name = uploaded_file.name
            st.session_state.page_view = "dashboard"
            st.rerun()

        data_json = st.session_state.get("data_json")
        ranking_df = st.session_state.get("ranking_df")
        if "best_cif_path" not in st.session_state:
            st.session_state.best_cif_path = st.session_state.parser.get_best_cif_path()

        best_cif_path = st.session_state.best_cif_path

        # ── 顶部指标条 ──
        total_models = len(ranking_df) if ranking_df is not None else 0
        best_score = ranking_df["ranking_score"].max() if total_models > 0 else 0.0
        # 项目名称：支持自定义编辑
        if "custom_project_name" not in st.session_state:
            default_name = data_json.get("name", "未命名") if data_json else "未命名"
            st.session_state.custom_project_name = default_name

        job_name = st.session_state.custom_project_name

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card"><div class="label">🧬 结构构象总数</div>
            <div class="value">{total_models}</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card"><div class="label">🏆 最高置信度</div>
            <div class="value">{best_score:.4f}</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-card"><div class="label">📌 项目名称</div>', unsafe_allow_html=True)
            new_name = st.text_input("项目名称", value=job_name, key="project_name_input",
                                     label_visibility="collapsed")
            if new_name != st.session_state.custom_project_name:
                st.session_state.custom_project_name = new_name
            st.markdown('</div>', unsafe_allow_html=True)
        with m4:
            cached = CacheManager.get_analyzed_count()
            st.markdown(f"""<div class="metric-card"><div class="label">💾 已分析工具</div>
            <div class="value">{cached} / {len(all_tools)}</div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ════════════════════════════════════════
        # 6. 视图路由
        # ════════════════════════════════════════

        # ── 仪表盘视图 ──
        if st.session_state.page_view == "dashboard":
            st.markdown("## ✨ AlphaFold 3 作业全景仪表盘")
            st.caption("选择下方工具卡片开始分析，已分析的工具将标记为绿色。")

            # 3D + 排行榜 tabs
            tab1, tab2 = st.tabs(["🔮 3D 结构探查", "🏅 模型排行榜"])
            with tab1:
                try:
                    if ("pocket_result" in st.session_state and
                            st.session_state.pocket_result.get("center") is not None):
                        pc = st.session_state.pocket_result["center"]
                        pp = st.session_state.pocket_result.get("pocket_points", None)
                        protein_view = render_protein_with_pocket(
                            best_cif_path, pocket_center=pc, pocket_radius=5.0,
                            pocket_color='magenta', pocket_points=pp)
                    else:
                        protein_view = render_protein(best_cif_path)
                    showmol(protein_view, height=500, width=800)
                except Exception as e3d:
                    st.warning(f"3D 渲染异常: {str(e3d)}")

            with tab2:
                if ranking_df is not None and not ranking_df.empty:
                    styled = ranking_df.style.background_gradient(subset=["ranking_score"], cmap="YlOrRd")
                    st.dataframe(styled, use_container_width=True, hide_index=True)
                else:
                    st.info("未检索到排行榜数据。")

            st.markdown("---")

            # ── 工具分类卡片 ──
            displayed_cats = [c for c in cfg["categories"]
                              if get_category_tools(c["id"])]

            for cat in displayed_cats:
                cat_tools = get_category_tools(cat["id"])
                color = cat.get("color", "#3b82f6")

                st.markdown(f"""
                <div class="category-header">
                    <span class="cat-icon">{cat['icon']}</span>
                    <span class="cat-name">{cat['name']}</span>
                    <span class="cat-desc">— {cat['desc']}</span>
                </div>
                """, unsafe_allow_html=True)

                cols = st.columns(3)
                for i, tool in enumerate(cat_tools):
                    with cols[i % 3]:
                        status = _tool_status(tool["id"])
                        badge_html = ('<span class="badge badge-analyzed">✅ 已分析</span>'
                                      if status == "analyzed"
                                      else '<span class="badge badge-unused">⬜ 未使用</span>')
                        is_from_dashboard = st.session_state.page_view != "tool"

                        st.markdown(f"""
                        <div class="tool-card" style="border-left:3px solid {color};">
                            <div class="icon">{tool['icon']}</div>
                            <div class="name">{tool['name']}</div>
                            <div class="desc">{tool['desc']}</div>
                            <div style="margin-top:0.5rem;">{badge_html}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        if st.button("▶ 打开", key=f"card_{tool['id']}", use_container_width=True):
                            st.session_state.selected_tool = tool["id"]
                            st.session_state.page_view = "tool"
                            st.rerun()

            st.markdown("---")
            st.caption("💡 提示: 点击工具卡片进入详细分析页面。分析完成后，数据将自动缓存供 AI 综合报告使用。")

        # ── 工具视图 ──
        elif st.session_state.page_view == "tool":
            selected_tool = st.session_state.get("selected_tool")
            tool_info = all_tools.get(selected_tool, {})

            if selected_tool is None:
                st.info("请从仪表盘选择一个工具开始分析。")
            elif selected_tool not in all_tools:
                st.warning(f"未知工具: {selected_tool}")
            else:
                # ═══════════════════════════════
                # === 全局基础分析 (统一模块) ===
                if selected_tool == "global_basic":
                    st.title("🧬 全局基础分析模块")
                    st.markdown("> 一键运行质量检查、湿实验性质评估、成药口袋探测，获得全景基础分析报告。")

                    if st.button("🚀 开始全局基础分析", type="primary", use_container_width=True):
                        results = {}

                        # 1) Basic QC
                        with st.spinner("① 质量检查 — 解析 pLDDT..."):
                            try:
                                from modules.parser import CifParser as CP
                                cp = CP(best_cif_path)
                                stats = cp.plddt_stats
                                metadata = cp.metadata
                                results["basic_qc"] = {"stats": stats, "metadata": metadata}
                                CacheManager.set("basic_qc", results["basic_qc"],
                                                 {"label": "基础质量检查", "summary": f"长度={metadata.get('length','?')}aa"})
                                del cp; gc.collect()
                            except Exception as e:
                                results["basic_qc"] = {"error": str(e)}

                        # 2) Wet-Lab Profiler
                        with st.spinner("② 湿实验评估 — 计算理化参数..."):
                            try:
                                from modules.wetlab_plugin import BiochemProfiler
                                profiler = BiochemProfiler(best_cif_path)
                                if profiler.error:
                                    results["wetlab"] = {"error": profiler.error}
                                else:
                                    results["wetlab"] = {"metrics": profiler.metrics, "sequence_length": len(profiler.sequence) if profiler.sequence else 0}
                                    CacheManager.set("wetlab_profiler", results["wetlab"],
                                                     {"label": "湿实验性质评估", "summary": f"MW={profiler.metrics.get('MW',0):.0f}Da"})
                                del profiler; gc.collect()
                            except Exception as e:
                                results["wetlab"] = {"error": str(e)}

                        # 3) Pocket Detection
                        with st.spinner("③ 口袋探测 — 网格扫描..."):
                            try:
                                df = safe_read_mmcif(best_cif_path).df['ATOM']
                                valid_atoms = df[df['type_symbol'] != 'H']
                                coords = valid_atoms[['Cartn_x', 'Cartn_y', 'Cartn_z']].values
                                res = PocketDetector.detect(valid_atoms, coords, 2.0, 1.5, 4.0)
                                st.session_state.pocket_result = res
                                results["pocket"] = {
                                    "volume": res["volume"], "hydrophobicity": res["hydrophobicity"],
                                    "residues": list(res["residues"]),
                                    "center": [float(res["center"][0]), float(res["center"][1]), float(res["center"][2])],
                                    "pocket_points_count": len(res.get("pocket_points", []))
                                }
                                CacheManager.set("pocket_detect", results["pocket"],
                                                 {"label": "成药口袋探测", "summary": f"体积={res['volume']:.0f}A3"})
                                del df, valid_atoms, coords; gc.collect()
                            except Exception as e:
                                results["pocket"] = {"error": str(e)}

                        CacheManager.set("global_basic", results,
                                         {"label": "全局基础分析", "summary": f"{len(results)}项完成"})
                        st.success("✅ 全局基础分析完成！")
                        st.rerun()

                    # 展示已缓存的全部分析结果
                    cached = CacheManager.get("global_basic")
                    if cached:
                        st.markdown("---")
                        st.subheader("📊 分析结果总览")

                        # QC
                        qc_data = cached.get("basic_qc", {})
                        if qc_data and "error" not in qc_data:
                            stats = qc_data.get("stats", {})
                            if stats:
                                st.markdown("**📊 基础质量检查 (pLDDT)**")
                                qc1, qc2, qc3, qc4 = st.columns(4)
                                qc1.metric("极高可信 (>90)", stats.get('>90', 0))
                                qc2.metric("高可信 (70-90)", stats.get('70-90', 0))
                                qc3.metric("低可信 (50-70)", stats.get('50-70', 0))
                                qc4.metric("极低可信 (<50)", stats.get('<50', 0))
                                fig = px.pie(values=list(stats.values()), names=[">90","70-90","50-70","<50"],
                                             title="pLDDT 分布", color_discrete_sequence=["#10b981","#f59e0b","#f97316","#ef4444"])
                                st.plotly_chart(fig, use_container_width=True)
                        elif qc_data.get("error"):
                            st.warning(f"QC 异常: {qc_data['error']}")

                        # Wetlab
                        wetlab = cached.get("wetlab", {})
                        if wetlab and "error" not in wetlab:
                            mets = wetlab.get("metrics", {})
                            st.markdown("**🧪 湿实验性质评估**")
                            w1, w2, w3 = st.columns(3)
                            w1.metric("分子量", f"{mets.get('MW', 0):.2f} Da")
                            w2.metric("等电点 (pI)", f"{mets.get('pI', 0):.2f}")
                            inst = mets.get('Instability_Index', 0)
                            w3.metric("不稳定指数", f"{inst:.2f}",
                                      delta="⚠ 不稳定" if inst > 40 else "✅ 稳定",
                                      delta_color="inverse" if inst > 40 else "normal")
                        elif wetlab.get("error"):
                            st.warning(f"湿实验评估异常: {wetlab['error']}")

                        # Pocket
                        pocket_data = cached.get("pocket", {})
                        if pocket_data and "error" not in pocket_data:
                            st.markdown("**🔍 成药口袋探测**")
                            p1, p2, p3 = st.columns(3)
                            p1.metric("口袋体积", f"{pocket_data['volume']:.1f} A3")
                            p2.metric("疏水性 GRAVY", f"{pocket_data['hydrophobicity']:.2f}")
                            p3.metric("网格点", pocket_data["pocket_points_count"])
                            if "pocket_result" in st.session_state:
                                try:
                                    protein_view = render_protein_with_pocket(
                                        best_cif_path,
                                        pocket_center=st.session_state.pocket_result["center"],
                                        pocket_radius=5.0, pocket_color='magenta',
                                        pocket_points=st.session_state.pocket_result.get("pocket_points"))
                                    showmol(protein_view, height=450, width=750)
                                except Exception:
                                    pass
                        elif pocket_data.get("error"):
                            st.warning(f"口袋探测异常: {pocket_data['error']}")

                # === 基础质量检查 ===
                elif selected_tool == "basic_qc":
                    st.title("📊 基础质量检查 (Basic QC)")
                    st.markdown("> 解析 mmCIF 结构文件，提取 pLDDT 置信度分数并进行分级统计。")

                    cached = CacheManager.get("basic_qc")
                    auto_run = cached is None

                    if auto_run or st.button("🔄 重新分析", use_container_width=True):
                        with st.spinner("正在解析结构并计算 pLDDT 质量分布..."):
                            from modules.parser import CifParser as CP
                            cp = CP(best_cif_path)
                            stats = cp.plddt_stats
                            metadata = cp.metadata
                            cache_data = {"stats": stats, "metadata": metadata}
                            CacheManager.set("basic_qc", cache_data,
                                             {"label": "基础质量检查", "summary": f"pLDDT >90: {stats.get('>90',0)}残基"})
                            cached = cache_data
                            del cp; gc.collect()

                    if cached:
                        stats = cached.get("stats", {})
                        metadata = cached.get("metadata", {})
                        if stats:
                            qc1, qc2, qc3, qc4 = st.columns(4)
                            qc1.metric("极高可信 (>90)", stats.get('>90', 0))
                            qc2.metric("高可信 (70-90)", stats.get('70-90', 0))
                            qc3.metric("低可信 (50-70)", stats.get('50-70', 0))
                            qc4.metric("极低可信 (<50)", stats.get('<50', 0))
                            fig = px.pie(values=list(stats.values()), names=[">90","70-90","50-70","<50"],
                                         title="pLDDT 置信度分布",
                                         color_discrete_sequence=["#10b981","#f59e0b","#f97316","#ef4444"])
                            st.plotly_chart(fig, use_container_width=True)
                            st.info(f"**蛋白质长度**: {metadata.get('length','N/A')} 残基 | **实体数**: {metadata.get('entity_count','N/A')}")
                        else:
                            st.warning("未能提取到有效的 pLDDT 数据。")
                    else:
                        st.warning("分析失败，请重试。")

                # === 成药口袋探测 ===
                elif selected_tool == "pocket_detect":
                    st.title("🔍 成药口袋探测 (Drug Pocket Detector)")
                    st.markdown("> 基于网格法的蛋白质表面成药结合口袋探测与 3D 可视化。")

                    # ── 参数持久化：从缓存恢复 ──
                    cached_params = CacheManager.get_params("pocket_detect")
                    col_ctrl, col_viz = st.columns([1, 2])
                    with col_ctrl:
                        st.subheader("⚙️ 探测参数")
                        grid_res = st.slider("网格分辨率 (A)", 0.5, 3.0,
                                             cached_params.get("grid_res", 2.0), 0.1, key="pd_grid")
                        min_dist = st.slider("最小表面距离 (A)", 0.5, 2.5,
                                             cached_params.get("min_dist", 1.5), 0.1, key="pd_min")
                        max_dist = st.slider("最大表面距离 (A)", 2.5, 8.0,
                                             cached_params.get("max_dist", 4.0), 0.1, key="pd_max")
                        color_options = ["magenta","cyan","yellow","lime","orange"]
                        probe_color = st.selectbox("标记颜色", color_options,
                                                   index=cached_params.get("color_idx", 0))
                        # ── 保存参数到缓存 ──
                        color_idx = color_options.index(probe_color)
                        CacheManager.save_params("pocket_detect", {
                            "grid_res": grid_res, "min_dist": min_dist,
                            "max_dist": max_dist, "color_idx": color_idx
                        })
                        detect_btn = st.button("🔍 开始检测", type="primary", use_container_width=True)

                    with col_viz:
                        if detect_btn:
                            with st.spinner("构建空间网格并计算口袋特征..."):
                                try:
                                    df = safe_read_mmcif(best_cif_path).df['ATOM']
                                    valid_atoms = df[df['type_symbol'] != 'H']
                                    coords = valid_atoms[['Cartn_x','Cartn_y','Cartn_z']].values
                                    res = PocketDetector.detect(valid_atoms, coords, grid_res, min_dist, max_dist)
                                    st.session_state.pocket_result = res
                                    cache_data = {
                                        "volume": res["volume"], "hydrophobicity": res["hydrophobicity"],
                                        "residues": list(res["residues"]),
                                        "center": [float(res["center"][0]), float(res["center"][1]), float(res["center"][2])],
                                        "pocket_points_count": len(res.get("pocket_points", [])),
                                        "params": {"grid_res": grid_res, "min_dist": min_dist,
                                                   "max_dist": max_dist, "color": probe_color}
                                    }
                                    CacheManager.set("pocket_detect", cache_data,
                                                     {"label": "成药口袋探测", "summary": f"体积={res['volume']:.0f}A3"})
                                    protein_view = render_protein_with_pocket(
                                        best_cif_path, pocket_center=res["center"], pocket_radius=5.0,
                                        pocket_color=probe_color, pocket_points=res.get("pocket_points"))
                                    st.subheader("🔮 口袋 3D 可视化")
                                    showmol(protein_view, height=500, width=750)
                                    st.caption(f"半透明 **{probe_color}** 小球 = 口袋网格点 | 大球 = 中心 (~5A)")
                                    del df, valid_atoms, coords; gc.collect()
                                except Exception as e:
                                    st.error(f"口袋探测异常: {str(e)}")
                        elif ("pocket_result" in st.session_state and
                              st.session_state.pocket_result.get("center") is not None):
                            res = st.session_state.pocket_result
                            protein_view = render_protein_with_pocket(
                                best_cif_path, pocket_center=res["center"], pocket_radius=5.0,
                                pocket_color=probe_color, pocket_points=res.get("pocket_points"))
                            st.subheader("🔮 口袋 3D 可视化 (缓存)")
                            showmol(protein_view, height=500, width=750)
                        else:
                            st.info("👈 调整参数后点击「开始检测」按钮。")

                    if "pocket_result" in st.session_state and st.session_state.pocket_result.get("center") is not None:
                        res = st.session_state.pocket_result
                        st.markdown("---")
                        st.subheader("📊 口袋特征数据")
                        pm1, pm2, pm3, pm4 = st.columns(4)
                        pm1.metric("口袋体积", f"{res['volume']:.1f} A3",
                                   delta="适合结合" if res['volume']>200 else "偏小",
                                   delta_color="normal" if res['volume']>200 else "inverse")
                        pm2.metric("疏水性 GRAVY", f"{res['hydrophobicity']:.2f}",
                                   delta="疏水" if res['hydrophobicity']>0 else "亲水",
                                   delta_color="normal" if res['hydrophobicity']>0 else "inverse")
                        pm3.metric("网格点", len(res.get('pocket_points', [])))
                        pm4.metric("中心坐标", f"({float(res['center'][0]):.1f}, {float(res['center'][1]):.1f}, {float(res['center'][2]):.1f})")
                        st.write("**口袋内衬残基 (距中心 5A):**")
                        st.code(", ".join(res['residues']) if len(res['residues'])>0 else "None")

                # === 湿实验性质评估 ===
                elif selected_tool == "wetlab_profiler":
                    st.title("🧪 湿实验性质评估 (Wet-Lab Profiler)")
                    st.markdown("> 评估靶点蛋白在体外表达纯化实验中的基本理化性质。")
                    with st.spinner("提取序列并计算理化参数..."):
                        from modules.wetlab_plugin import BiochemProfiler
                        profiler = BiochemProfiler(best_cif_path)
                        if profiler.error:
                            st.error(profiler.error)
                        else:
                            mets = profiler.metrics
                            CacheManager.set("wetlab_profiler",
                                             {"metrics": mets, "sequence_length": len(profiler.sequence) if profiler.sequence else 0},
                                             {"label": "湿实验性质评估", "summary": f"MW={mets.get('MW',0):.0f}Da"})
                            w1, w2, w3 = st.columns(3)
                            w1.metric("分子量", f"{mets.get('MW',0):.2f} Da")
                            w2.metric("等电点 (pI)", f"{mets.get('pI',0):.2f}")
                            inst = mets.get('Instability_Index',0)
                            w3.metric("不稳定指数", f"{inst:.2f}",
                                      delta="⚠ 不稳定" if inst>40 else "✅ 稳定",
                                      delta_color="inverse" if inst>40 else "normal")
                            if inst > 40:
                                st.error(f"⚠️ 不稳定指数 {inst:.2f} > 40，该蛋白在体外表达时可能不稳定。")
                            else:
                                st.success(f"✅ 不稳定指数 {inst:.2f} ≤ 40，蛋白在标准体外条件下预期稳定。")
                            if profiler.sequence:
                                with st.expander("🔍 氨基酸序列", expanded=False):
                                    st.code(profiler.sequence[:500] + ("..." if len(profiler.sequence)>500 else ""))
                        del profiler; gc.collect()

                # === fpocket 口袋探测 ===
                elif selected_tool == "fpocket_detect":
                    st.title("🎯 fpocket 成药结合口袋探测")
                    st.markdown("> 调用 fpocket C 引擎进行高精度几何口袋扫描。")
                    with st.spinner("调用 fpocket 引擎..."):
                        from modules.fpocket_plugin import FpocketRunner
                        runner = FpocketRunner(best_cif_path)
                        success = runner.run()
                        if runner.error:
                            st.warning(f"fpocket: {runner.error}")
                            st.info("替代方案: 使用「成药口袋探测」网格法，或 CASTp 在线桥接。")
                        if success and runner.top_pockets:
                            st.success(f"✅ 探测到 {len(runner.top_pockets)} 个候选口袋！")
                            df_pockets = runner.get_results_df()
                            if not df_pockets.empty:
                                st.dataframe(df_pockets, use_container_width=True)
                            CacheManager.set("fpocket_detect", runner.top_pockets,
                                             {"label": "fpocket 口袋探测", "summary": f"{len(runner.top_pockets)}个口袋"})
                            if runner.pocket_pdbs:
                                st.subheader("🔮 口袋 3D 可视化")
                                cif_data = st.session_state.get("cif_content", "")
                                if not cif_data:
                                    with open(best_cif_path, "r") as f: cif_data = f.read()
                                view = py3Dmol.view(width=800, height=500)
                                view.addModel(cif_data, "cif")
                                view.setStyle({'model':0}, {'cartoon':{'colorscheme':{'prop':'b','gradient':'roygb','min':50,'max':100}}})
                                colors = ['magenta','yellow','cyan']
                                for i, (pn, pd_) in enumerate(runner.pocket_pdbs.items()):
                                    view.addModel(pd_, 'pdb')
                                    view.setStyle({'model':-1}, {'sphere':{'color':colors[i%3],'alpha':0.5}})
                                view.zoomTo(); showmol(view, height=500, width=800)
                        elif not runner.error:
                            st.info("未探测到显著口袋。")
                        del runner; gc.collect()

                # === Foldseek ===
                elif selected_tool == "foldseek_search":
                    st.title("🔬 Foldseek 结构相似性检索")
                    st.markdown("> 将本地结构发送至 EBI Foldseek 服务器（自动 CIF→PDB 转换），在 PDB 全库中检索结构同源蛋白。")
                    if st.button("🌐 开始 Foldseek 全库搜索", type="primary", use_container_width=True):
                        with st.spinner("转换格式并上传至 Foldseek 服务器..."):
                            from modules.foldseek_plugin import FoldseekAPIWrapper
                            wrapper = FoldseekAPIWrapper(best_cif_path)
                            ok = wrapper.run()
                            if wrapper.error:
                                st.error(f"Foldseek: {wrapper.error}")
                                # 技术诊断
                                with st.expander("🔍 技术诊断", expanded=False):
                                    st.caption("已自动将 CIF 转换为 PDB 格式后上传。")
                                    st.caption("如持续无匹配，可尝试 CASTp 在线桥接或 PAE 分析。")
                            elif ok:
                                df_results = wrapper.get_results_df()
                                if not df_results.empty:
                                    CacheManager.set("foldseek_search", df_results.to_dict("records"),
                                                     {"label": "Foldseek", "summary": f"{len(df_results)}个匹配"})
                                    st.success(f"✅ 匹配 {len(df_results)} 个结构！")
                                    md = "|PDB ID|TM-score|Seq Identity|E-value|Aln Len|\n|:---|:---|:---|:---|:---|\n"
                                    for _, r in df_results.iterrows():
                                        pdb = r.get('PDB ID', 'N/A')
                                        md += (f"|`{pdb}`|{r.get('TM-score',0):.4f}|{r.get('Sequence Identity',0):.4f}|"
                                               f"{r.get('E-value','N/A')}|{r.get('Alignment Length',0)}|\n")
                                    st.markdown(md)
                                else:
                                    st.info("未返回有效结果。可能原因：1) 蛋白结构新颖 2) 格式兼容性。"
                                            "已自动将 CIF 转为标准 PDB 格式。")
                            del wrapper; gc.collect()

                # === PAE ===
                elif selected_tool == "pae_analysis":
                    st.title("📐 PAE 空间位置误差矩阵分析")
                    st.markdown("> 可视化 AlphaFold 3 预测的空间位置误差热力图。")
                    with st.spinner("定位 PAE JSON 并解析矩阵..."):
                        from modules.pae_plugin import PaeVisualizer
                        viz = PaeVisualizer(best_cif_path)
                        if viz.error:
                            st.warning(f"PAE: {viz.error}")
                        elif viz.pae_matrix is not None:
                            pae_arr = np.array(viz.pae_matrix)
                            CacheManager.set("pae_analysis",
                                             {"shape": list(pae_arr.shape), "mean": float(pae_arr.mean()),
                                              "median": float(np.median(pae_arr)), "max": float(pae_arr.max())},
                                             {"label": "PAE 分析", "summary": f"均值={pae_arr.mean():.1f}A"})
                            st.success(f"✅ PAE 矩阵: {pae_arr.shape}")
                            fig = px.imshow(pae_arr, color_continuous_scale="Blues_r", zmin=0, zmax=30,
                                            labels=dict(x="Scored Residue", y="Aligned Residue", color="PAE (A)"),
                                            title="Predicted Aligned Error (PAE)")
                            fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
                            st.plotly_chart(fig, use_container_width=True)
                            a1, a2, a3 = st.columns(3)
                            a1.metric("均值 PAE", f"{pae_arr.mean():.2f} A")
                            a2.metric("中位数 PAE", f"{np.median(pae_arr):.2f} A")
                            a3.metric("最大值 PAE", f"{pae_arr.max():.2f} A")
                        del viz; gc.collect()

                # === Pfam ===
                elif selected_tool == "pfam_annotator":
                    st.title("🏷 Pfam 蛋白结构域注释")
                    st.markdown("> 通过 EBI HMMER API 进行 Pfam 结构域快速注释（自动适配多版本 API）。")
                    if st.button("🔬 开始 Pfam 结构域注释", type="primary", use_container_width=True):
                        from modules.domain_plugin import DomainAnnotator
                        from utils.visualization import render_domain_timeline

                        # 进度显示区域
                        progress_bar = st.progress(0, text="准备中...")
                        status_text = st.empty()

                        def _pfam_progress(status: str, progress: float):
                            """进度回调：更新 Streamlit 进度条和状态文本"""
                            try:
                                progress_bar.progress(min(max(progress, 0.0), 1.0), text=status)
                                status_text.caption(status)
                            except Exception:
                                pass

                        annotator = DomainAnnotator(best_cif_path)
                        if annotator.error:
                            st.error(annotator.error)
                            progress_bar.empty(); status_text.empty()
                        else:
                            api_ok = annotator.run_annotation(progress_callback=_pfam_progress)
                            if annotator.error:
                                st.warning(f"HMMER API: {annotator.error}")
                                # 显示调试信息
                                debug = annotator.get_debug_info()
                                if debug:
                                    with st.expander("🔍 API 调试日志", expanded=False):
                                        for d in debug:
                                            st.caption(f"• {d}")
                            elif api_ok:
                                df_domains = annotator.get_results_df()
                                if not df_domains.empty:
                                    cache_data = {
                                        "domains": df_domains.to_dict("records"),
                                        "seq_length": annotator.seq_length
                                    }
                                    CacheManager.set("pfam_annotator", cache_data,
                                                     {"label": "Pfam 注释", "summary": f"{len(df_domains)}个结构域"})
                                    st.success(f"✅ 注释到 {len(df_domains)} 个结构域！")
                                    st.dataframe(df_domains, use_container_width=True)
                                    try:
                                        fig_d = render_domain_timeline(df_domains, annotator.seq_length)
                                        st.plotly_chart(fig_d, use_container_width=True)
                                    except Exception as ve:
                                        st.warning(f"可视化异常: {ve}")
                                else:
                                    st.info("未检测到已知 Pfam 结构域。")
                                    debug = annotator.get_debug_info()
                                    if debug:
                                        with st.expander("🔍 API 调试日志", expanded=False):
                                            for d in debug:
                                                st.caption(f"• {d}")
                            else:
                                st.info("API 调用未返回错误，但也未找到结构域。")
                            progress_bar.empty(); status_text.empty()
                            del annotator; gc.collect()

                # === AI 综合报告 ===
                elif selected_tool == "ai_report":
                    st.title("🤖 AI 一键综合报告生成")
                    st.markdown("> 汇总所有已分析工具的数据与参数配置，借助大语言模型生成完整结构生物学洞察报告。")

                    all_cached = CacheManager.get_all()
                    all_params = {tid: CacheManager.get_params(tid) for tid in all_cached}
                    all_meta = CacheManager.get_all_meta()

                    if all_cached:
                        st.subheader(f"📋 已累积分析数据 ({len(all_cached)} 个工具)")
                        for tid, data in all_cached.items():
                            meta = all_meta.get(tid, {})
                            params = all_params.get(tid, {})
                            label = meta.get('label', tid)
                            summary = meta.get('summary', '')
                            with st.expander(f"📌 {label} — {summary}", expanded=False):
                                st.caption(f"**缓存时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(meta.get('timestamp', 0)))}")
                                if params:
                                    st.caption(f"**分析参数**: {json.dumps(params, ensure_ascii=False)}")
                                st.json(data)
                    else:
                        st.info("💡 尚未运行分析模块。系统将仅使用基础置信度数据。建议先运行至少一个模块。")

                    col_r1, col_r2 = st.columns([2, 1])
                    with col_r1:
                        if st.button("🚀 生成 AI 综合深度解读报告", type="primary", use_container_width=True):
                            if not api_key:
                                st.warning("⚠️ 请先在侧边栏填写 API Key。")
                            else:
                                with st.spinner("聚合分析数据并连线 AI 引擎..."):
                                    from modules.report_generator import LLMReportBuilder
                                    builder = LLMReportBuilder(api_key, api_base, model_name)
                                    # 组合数据 + 参数 + 元信息
                                    context_data = {
                                        "project_name": job_name,
                                        "total_models": total_models,
                                        "best_confidence": best_score,
                                        "analyzed_tools": all_cached,
                                        "tool_params": all_params,
                                        "tool_meta": all_meta
                                    }
                                    try:
                                        base_conf = st.session_state.parser.extract_confidences()
                                        context_data["base_confidences"] = base_conf
                                    except Exception as conf_err:
                                        print(f"[WARN] extract_confidences failed: {conf_err}", file=__import__('sys').stderr)
                                    st.session_state.full_context = context_data
                                st.success("✨ AI 专家级综合报告流式生成中：")
                                report_placeholder = st.empty()
                                full_text = ""
                                for chunk in builder.generate_stream(context_data):
                                    full_text += chunk
                                    report_placeholder.markdown(full_text)
                                st.session_state.full_report_text = full_text
                                CacheManager.set("ai_report",
                                                 {"report": full_text, "context": context_data},
                                                 {"label": "AI 综合报告", "summary": "报告已生成"})
                                del builder; gc.collect()
                    with col_r2:
                        if "full_report_text" in st.session_state and st.session_state.full_report_text:
                            final_md = MarkdownExporter.generate_report_markdown(
                                st.session_state.get("full_context", {}), st.session_state.full_report_text)
                            st.download_button("📥 下载 Markdown 报告", data=final_md,
                                               file_name=f"{job_name}_comprehensive_report.md",
                                               mime="text/markdown", use_container_width=True)

                # === UniProt ===
                elif selected_tool == "uniprot_annotator":
                    st.title("🌐 UniProt 蛋白功能注释检索")
                    st.markdown("> 通过 UniProt REST API 检索蛋白功能注释、GO 术语、亚细胞定位等。")
                    uniprot_id = st.text_input("UniProt ID (如 P00533):", placeholder="例如: P00533")
                    if st.button("🔍 检索 UniProt 功能注释", type="primary", use_container_width=True):
                        import requests
                        with st.spinner("查询 UniProt REST API..."):
                            try:
                                url = None
                                if uniprot_id:
                                    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
                                else:
                                    jc = job_name.split("_")[0] if "_" in job_name else job_name
                                    r = requests.get(f"https://rest.uniprot.org/uniprotkb/search?query={jc}&size=1&format=json", timeout=15)
                                    results = r.json().get("results", [])
                                    if results:
                                        uniprot_id = results[0].get("primaryAccession", "")
                                        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
                                    else:
                                        st.warning("未能自动匹配，请手动输入 UniProt ID。")
                                if url:
                                    resp = requests.get(url, timeout=15)
                                    data = resp.json()
                                    CacheManager.set("uniprot_annotator", data,
                                                     {"label": "UniProt 注释", "summary": data.get('primaryAccession','?')})
                                    st.success(f"✅ **{data.get('primaryAccession','N/A')}** — "
                                               f"{data.get('proteinDescription',{}).get('recommendedName',{}).get('fullName',{}).get('value','N/A')}")
                                    cu1, cu2 = st.columns(2)
                                    cu1.metric("序列长度", data.get('sequence',{}).get('length','?'))
                                    with cu2:
                                        st.metric("物种", data.get('organism',{}).get('scientificName','?'))
                                    go_terms = []
                                    for ref in data.get('uniProtKBCrossReferences', []):
                                        if ref.get('database') == 'GO':
                                            for prop in ref.get('properties', []):
                                                if prop.get('key') == 'GoTerm':
                                                    go_terms.append(f"{ref.get('id')}: {prop.get('value')}")
                                    if go_terms:
                                        with st.expander(f"🏷 GO 注释 ({len(go_terms)} 条)", expanded=False):
                                            for gt in go_terms[:20]: st.markdown(f"- {gt}")
                            except Exception as e:
                                st.error(f"检索出错: {str(e)}")
                            finally:
                                gc.collect()

                # === CASTp / PrankWeb ===
                elif selected_tool == "castp_bridge":
                    st.title("🔗 蛋白口袋在线预测")
                    st.info(" **PrankWeb API** (全自动): [https://prankweb.cz/](https://prankweb.cz/) | **CASTp** (手动): [http://sts.bioe.uic.edu/castp/](http://sts.bioe.uic.edu/castp/)")

                    from modules.prankweb_plugin import PrankWebRunner

                    # 自动 PrankWeb 预测
                    st.markdown("#### 全自动口袋预测 (PrankWeb API)")
                    if st.button(" 自动预测口袋 (PrankWeb)", type="primary", use_container_width=True):
                        runner = PrankWebRunner(best_cif_path, job_name)
                        with st.spinner("正在提交结构到 PrankWeb 服务器并等待预测..."):
                            ok = runner.run()

                        if runner.error:
                            st.error(f"PrankWeb: {runner.error}")
                            st.info("PrankWeb API 不可用时，请使用下方 CASTp 手动桥接。")
                        elif ok:
                            pocket_data = runner.get_results_df_data()
                            if pocket_data:
                                st.success(f"✅ PrankWeb 预测完成！发现 {len(pocket_data)} 个候选口袋")
                                st.dataframe(pd.DataFrame(pocket_data), use_container_width=True, hide_index=True)
                                CacheManager.set("castp_bridge",
                                                 {"pockets": pocket_data, "source": "prankweb"},
                                                 {"label": "口袋预测", "summary": f"{len(pocket_data)} 个口袋 (PrankWeb)"})
                            else:
                                st.warning("PrankWeb 返回数据但未能解析口袋，请查看原始数据。")
                        runner.cleanup(); del runner; gc.collect()

                    # CASTp 手动桥接 (回退)
                    with st.expander("📤 CASTp 手动桥接（PrankWeb 不可用时）", expanded=False):
                        ti, to = st.tabs(["下载结构", "导入结果"])
                        with ti:
                            cif_content = st.session_state.get("cif_content", "")
                            if not cif_content:
                                with open(best_cif_path, "r") as f: cif_content = f.read()
                            st.download_button(" 下载 CIF (用于 CASTp 上传)", data=cif_content,
                                               file_name=f"{job_name}_for_castp.cif", mime="chemical/x-cif")
                        with to:
                            castp_file = st.file_uploader("上传 CASTp 结果 (JSON/TXT/ZIP):", type=["json","txt","zip"])
                            if castp_file:
                                raw_bytes = castp_file.read()
                                content = ""
                                fname = castp_file.name.lower()
                                castp_pockets = []
                                castp_bulb_data = None

                                if fname.endswith('.zip'):
                                    import zipfile, io, json as _json
                                    try:
                                        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                                            all_names = zf.namelist()
                                            poc_names = [n for n in all_names if n.lower().endswith('.poc') and 'mouth' not in n.lower()]
                                            if poc_names:
                                                poc_content = zf.read(poc_names[0]).decode("utf-8", errors="ignore")
                                                castp_pockets = _parse_castp_poc(poc_content)
                                            bulb_names = [n for n in all_names if 'bulb' in n.lower() and n.lower().endswith('.json')]
                                            if not bulb_names:
                                                bulb_names = [n for n in all_names if n.lower().endswith('.json') and 'source' not in n.lower()]
                                            if bulb_names:
                                                target = max(bulb_names, key=lambda n: zf.getinfo(n).file_size)
                                                castp_bulb_data = _json.loads(zf.read(target).decode("utf-8", errors="ignore"))
                                                content = f"[CASTp bulb JSON: {len(castp_bulb_data)} groups]"
                                            if not castp_bulb_data and not castp_pockets:
                                                json_names = [n for n in all_names if n.lower().endswith('.json')]
                                                if json_names:
                                                    target = max(json_names, key=lambda n: zf.getinfo(n).file_size)
                                                    content = zf.read(target).decode("utf-8", errors="ignore")
                                                else:
                                                    txt_names = [n for n in all_names if n.lower().endswith(('.txt', '.pocinfo'))]
                                                    if txt_names:
                                                        target = max(txt_names, key=lambda n: zf.getinfo(n).file_size)
                                                        content = zf.read(target).decode("utf-8", errors="ignore")
                                    except Exception as zip_err:
                                        st.error(f"ZIP 解析失败: {zip_err}")
                                else:
                                    content = raw_bytes.decode("utf-8", errors="ignore")
                                    import json as _json
                                    try:
                                        parsed = _json.loads(content)
                                        if isinstance(parsed, list):
                                            castp_bulb_data = parsed
                                    except Exception:
                                        castp_pockets = _parse_castp_poc(content)

                                if castp_pockets:
                                    st.success(f"✅ 解析到 {len(castp_pockets)} 个口袋！")
                                    st.dataframe(pd.DataFrame(castp_pockets), use_container_width=True, hide_index=True)
                                    CacheManager.set("castp_bridge",
                                                     {"pockets": castp_pockets, "bulb": castp_bulb_data, "source": "castp"},
                                                     {"label": "CASTp 结果", "summary": f"{len(castp_pockets)} 个口袋"})
                                elif castp_bulb_data:
                                    st.success("✅ CASTp 球体数据已加载！")
                                    st.info(f"共 {len(castp_bulb_data)} 组球体坐标")
                                    CacheManager.set("castp_bridge",
                                                     {"pockets": [], "bulb": castp_bulb_data, "source": "castp"},
                                                     {"label": "CASTp 结果", "summary": f"{len(castp_bulb_data)} 组球体"})
                                elif content:
                                    st.success("✅ 文件已接收！")
                                    with st.expander(" 结果预览", expanded=False):
                                        st.text(content[:5000])
                                    CacheManager.set("castp_bridge", {"raw": content[:10000], "source": "castp"},
                                                     {"label": "CASTp 结果", "summary": "已导入"})
                                else:
                                    st.error("未能从文件中提取有效数据")

                # === iCn3D ===
                elif selected_tool == "icn3d_bridge":
                    st.title("🔮 iCn3D 三维结构可视化")
                    st.info(" **iCn3D**: [https://www.ncbi.nlm.nih.gov/Structure/icn3d/](https://www.ncbi.nlm.nih.gov/Structure/icn3d/)")

                    # 尝试从 UniProt 缓存获取 ID，用于自动生成 iCn3D 链接
                    uniprot_cache = CacheManager.get("uniprot_annotator")
                    uniprot_id = ""
                    if uniprot_cache and isinstance(uniprot_cache, dict):
                        uniprot_id = uniprot_cache.get("primaryAccession", "")

                    if uniprot_id:
                        st.success(f"✅ 检测到 UniProt ID: **{uniprot_id}**，可自动加载 AlphaFold 结构")

                        # 生成 iCn3D 直接查看 URL
                        icn3d_url = f"https://www.ncbi.nlm.nih.gov/Structure/icn3d/?afid={uniprot_id}&width=600&height=400&showcommand=0&mobilemenu=1&showtitle=0"

                        col_i1, col_i2 = st.columns([3, 1])
                        with col_i1:
                            st.markdown("#### 在线 3D 查看 (iCn3D 嵌入)")
                            st.components.v1.iframe(icn3d_url, width=640, height=440, scrolling=False)
                        with col_i2:
                            st.markdown("#### 快捷操作")
                            st.markdown(f"[🔗 全屏打开]({icn3d_url})")
                            st.caption("pLDDT 着色方案：")
                            st.code("color #0053d6 if bfactor > 90\ncolor #65cbf3 if bfactor > 70\ncolor #fddb00 if bfactor > 50\ncolor #ff7d45 if bfactor <= 50", language="text")
                    else:
                        st.info("💡 先运行「UniProt 功能注释」模块获取 UniProt ID，即可自动嵌入 iCn3D 查看器。")

                    # 自定义结构手动导入 (回退)
                    with st.expander("📤 自定义结构查看（无 UniProt ID 时）", expanded=False):
                        from modules.foldseek_plugin import FoldseekAPIWrapper
                        wrapper = FoldseekAPIWrapper(best_cif_path)
                        pdb_content, _ = wrapper._read_file(best_cif_path)
                        del wrapper; gc.collect()

                        st.download_button(" 下载 PDB (拖入 iCn3D)", data=pdb_content,
                                           file_name=f"{job_name}_for_icn3d.pdb", mime="chemical/x-pdb")
                        st.caption("下载后拖入 [iCn3D](https://www.ncbi.nlm.nih.gov/Structure/icn3d/) 的 File > Open File > PDB File")

                    # 结果导入
                    with st.expander("📥 导入 iCn3D 分析结果", expanded=False):
                        icn3d_file = st.file_uploader("上传 iCn3D 结果:", type=["pdb","png","json","cif","py"])
                        if icn3d_file:
                            content = icn3d_file.read()
                            file_name = icn3d_file.name
                            st.success(f"✅ 已接收文件: {file_name}")
                            if file_name.endswith('.png'):
                                st.image(content, caption="iCn3D 导出图像", use_container_width=True)
                            elif file_name.endswith(('.pdb', '.cif', '.py')):
                                text_content = content.decode("utf-8", errors="ignore")
                                with st.expander(" 结果预览", expanded=True):
                                    st.text(text_content[:5000])
                            else:
                                with st.expander(" 内容预览", expanded=True):
                                    try:
                                        text_content = content.decode("utf-8", errors="ignore")
                                        st.json(text_content[:5000]) if file_name.endswith('.json') else st.text(text_content[:3000])
                                    except:
                                        st.info("二进制文件，无法预览")
                            CacheManager.set("icn3d_bridge", {"file": file_name, "size": len(content)},
                                             {"label": "iCn3D 结果", "summary": f"已导入 {file_name}"})

                # === ESM Atlas ===
                elif selected_tool == "esm_atlas":
                    st.title("🧠 ESM Atlas 蛋白语言模型分析")
                    st.info("🌐 **ESM Atlas**: [https://esmatlas.com/](https://esmatlas.com/)")

                    from modules.esm_plugin import ESMFoldRunner
                    from modules.wetlab_plugin import BiochemProfiler

                    # 提取序列
                    profiler = BiochemProfiler(best_cif_path)
                    seq = profiler.sequence if not profiler.error else ""
                    del profiler; gc.collect()

                    if not seq:
                        st.warning("未能从 CIF 文件中提取有效序列。")
                    else:
                        st.success(f"✅ 已提取序列 ({len(seq)} aa)")

                        # 自动调用 ESMFold API
                        if st.button("▶ 自动折叠 (ESMFold API)", type="primary", use_container_width=True):
                            runner = ESMFoldRunner(seq, job_name)
                            with st.spinner("正在调用 ESM Atlas API 进行结构预测..."):
                                ok = runner.run()

                            if runner.error:
                                st.error(f"ESMFold: {runner.error}")
                            elif ok:
                                summary = runner.get_confidence_summary()
                                dist = summary.get("distribution", {})
                                st.success(
                                    f"✅ ESMFold 预测完成！"
                                    f" {summary['residue_count']} 残基, "
                                    f"平均 pLDDT={summary['mean_plddt']:.2f}"
                                    + (f", pTM={summary['ptm']:.3f}" if summary.get('ptm') else "")
                                )

                                # 置信度分布
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric(">90 极高可信", dist.get(">90", 0))
                                c2.metric("70-90 高可信", dist.get("70-90", 0))
                                c3.metric("50-70 低可信", dist.get("50-70", 0))
                                c4.metric("<50 极低可信", dist.get("<50", 0))

                                # 下载 PDB
                                pdb_data, pdb_fname = runner.get_pdb_download()
                                st.download_button(
                                    " 下载 ESMFold PDB",
                                    data=pdb_data,
                                    file_name=pdb_fname,
                                    mime="chemical/x-pdb"
                                )

                                # 缓存结果
                                CacheManager.set(
                                    "esm_atlas",
                                    {
                                        "pdb": pdb_data[:50000],
                                        "mean_plddt": summary["mean_plddt"],
                                        "ptm": summary.get("ptm"),
                                        "residue_count": summary["residue_count"],
                                        "distribution": dist,
                                    },
                                    {
                                        "label": "ESM 结果",
                                        "summary": f"pLDDT={summary['mean_plddt']:.1f}, {summary['residue_count']}aa"
                                    }
                                )
                            del runner; gc.collect()

                        # 手动上传回退（API 失败时使用）
                        with st.expander("📤 手动上传结果（API 不可用时）", expanded=False):
                            st.caption("下载序列 FASTA → 在 ESM Atlas 网站折叠 → 上传结果")
                            fasta = f">AF3_{job_name}\n{seq}"
                            st.download_button(" 下载 FASTA", data=fasta, file_name=f"{job_name}.fasta")
                            esm_file = st.file_uploader("上传 ESM 结果:", type=["pdb", "json", "txt"])
                            if esm_file:
                                content = esm_file.read().decode("utf-8", errors="ignore")
                                st.success("✅ 已接收！")
                                CacheManager.set("esm_atlas", content[:10000], {"label": "ESM 结果"})

                # === STRING PPI ===
                elif selected_tool == "string_ppi":
                    st.title(" STRING 蛋白互作网络")
                    st.info(" **STRING DB**: [https://string-db.org/](https://string-db.org/)")

                    from modules.string_plugin import STRINGQuery

                    # 自动填充蛋白名称
                    default_query = job_name if job_name != "N/A" else ""
                    # 尝试从 UniProt 缓存中获取基因名
                    uniprot_cache = CacheManager.get("uniprot_annotator")
                    if uniprot_cache and isinstance(uniprot_cache, dict):
                        gene_names = uniprot_cache.get("genes", [])
                        if gene_names:
                            default_query = gene_names[0].get("geneName", {}).get("value", default_query)

                    sq = st.text_input("蛋白名称或 UniProt ID:", value=default_query)

                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        score_threshold = st.select_slider("置信度阈值", options=[150, 400, 700, 900],
                                                            value=400, format_func=lambda x: {150:"低",400:"中",700:"高",900:"最高"}[x])
                    with col_s2:
                        add_nodes = st.slider("额外互作节点", 0, 10, 10)

                    if st.button(" 自动查询互作网络", type="primary", use_container_width=True):
                        if not sq or sq in ("未命名", "N/A", "unnamed"):
                            st.warning("请输入有效的蛋白名称或 UniProt ID（如 TP53、P04637）。")
                        else:
                            with st.spinner(f"正在查询 STRING API: {sq}..."):
                                query = STRINGQuery(sq)
                                ok = query.run(required_score=score_threshold, add_nodes=add_nodes)

                            if query.error:
                                st.error(f"STRING: {query.error}")
                            elif ok:
                                summary = query.get_summary()
                                st.success(
                                    f" **{summary['preferred_name']}** — "
                                    f"找到 {summary['interaction_count']} 条互作关系 "
                                    f"(最高得分: {summary['max_score']:.3f})"
                                )

                                # 互作网络表格
                                interactions = query.get_results()
                                if interactions:
                                    import pandas as _pd
                                    df_ppi = _pd.DataFrame(interactions)
                                    st.dataframe(df_ppi.head(30), use_container_width=True, hide_index=True)

                                    # 网络图
                                    st.markdown("#### 互作网络可视化")
                                    st.image(query.network_image_url, caption=f"STRING 互作网络 — {summary['preferred_name']}", use_container_width=True)

                                    # 缓存
                                    CacheManager.set("string_ppi", interactions[:50],
                                                     {"label": "STRING PPI", "summary": f"{len(interactions)}条互作"})
                                else:
                                    st.info("未找到互作关系，尝试降低置信度阈值。")
                            del query; gc.collect()

                    # 手动上传回退
                    with st.expander(" 手动上传 TSV（API 不可用时）", expanded=False):
                        string_file = st.file_uploader("上传 STRING TSV 互作数据:", type=["tsv","txt","csv"])
                        if string_file:
                            try:
                                df_ppi = pd.read_csv(string_file, sep="\t")
                                st.success(f"✅ {len(df_ppi)} 条互作关系")
                                st.dataframe(df_ppi.head(20), use_container_width=True)
                                CacheManager.set("string_ppi", df_ppi.to_dict("records")[:50],
                                                 {"label": "STRING PPI", "summary": f"{len(df_ppi)}条"})
                            except Exception as e:
                                st.error(f"解析失败: {e}")

                # === 未知 ===
                else:
                    st.info(f"💡 模块 '{selected_tool}' 已注册，请在仪表盘中选择工具开始分析。")

    except AF3ValidationError as ve:
        st.warning(f"💡 文件校验提示: {str(ve)}")
    except Exception as e:
        st.error(f"💥 系统运行异常: {str(e)}")
        import traceback
        with st.expander("🔍 错误详情", expanded=False):
            st.code(traceback.format_exc())

# ════════════════════════════════════════════════
# 7. 全局内存清理
# ════════════════════════════════════════════════
gc.collect()