import openai
import gc

# ═══════════════════════════════════════════════════════════════════
# AI 综合报告提示词 — 学术严谨 + 多维综合 + 逻辑清晰
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "你是一位资深结构生物学家与计算生物学专家，"
    "擅长将 AlphaFold 3 多模块分析数据转化为专业、严谨、逻辑清晰的结构生物学综合分析报告。\n\n"
    "【写作规范】\n"
    "1. 语言风格：学术化、专业化，使用结构生物学标准术语。"
    "避免口语化表达和过度拟人化比喻。\n"
    "2. 逻辑结构：每个结论必须有数据支撑。先陈述数据事实，再给出专业解读，最后说明生物学意义或实验建议。\n"
    "3. 数据综合：不要逐条罗列数据，要将不同模块的结果交叉印证、综合分析。"
    "例如将 pLDDT 分布与 PAE 矩阵结合评估模型可靠性，将 Pfam 结构域与 Foldseek 匹配结果结合推断功能。\n"
    "4. 篇幅控制：简洁精炼，每个章节聚焦核心发现，避免冗余重复。\n\n"
    "【数据隔离红线】\n"
    "严禁提及、联想或引用任何与 TCGA（癌症基因组图谱）相关的数据库内容、癌症队列、临床相关性或衍生术语。\n\n"
    "【输出格式】\n"
    "使用 Markdown 格式，包含以下章节结构（可根据实际数据可用性调整）：\n"
    "- ## 一、模型质量评估（pLDDT + pTM/ipTM + PAE 综合分析）\n"
    "- ## 二、蛋白理化性质与实验可行性（分子量 + pI + 不稳定指数）\n"
    "- ## 三、结构域组成与功能推断（Pfam 结构域 + Foldseek 同源比对综合解读）\n"
    "- ## 四、结合口袋与成药性分析（fpocket/网格法口袋探测结果）\n"
    "- ## 五、综合结论与后续实验建议"
)

USER_PROMPT_PREFIX = (
    "请基于以下 AlphaFold 3 多模块分析数据，生成一份专业、严谨、逻辑清晰的结构生物学综合分析报告。\n\n"
    "【数据上下文】\n"
)

USER_PROMPT_SUFFIX = (
    "\n\n【报告要求】\n"
    '1. 每个章节必须基于实际数据进行分析，数据缺失的模块简要说明即可，不要编造数据。\n'
    '2. 在"结构域组成与功能推断"章节中，综合 Pfam 结构域注释和 Foldseek 结构相似性检索结果，'
    '推断蛋白的可能功能分类和生物学角色。\n'
    '3. 在"结合口袋与成药性分析"章节中，对检测到的口袋按可成药性评分排序，评估其作为药物靶点的潜力。\n'
    '4. 在"综合结论"中，用 3-5 句话概括该蛋白的结构特征、功能推测和最具价值的后续研究方向。\n'
    '5. 直接输出报告正文，不要输出"好的""以下是"等开场白。'
)


class LLMReportBuilder:
    def __init__(self, api_key, base_url, model_name="gemini-3.1-pro"):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def build_prompt(self, context_data):
        prompt = USER_PROMPT_PREFIX

        has_data = False

        # v3.1 兼容：支持从 analyzed_tools (CacheManager 格式) 和旧版扁平键两种格式提取数据
        analyzed_tools = context_data.get("analyzed_tools", {})
        tool_meta = context_data.get("tool_meta", {})
        base_conf = context_data.get("base_confidences", {})

        # 项目基本信息
        prompt += f"- 项目名称: {context_data.get('project_name', '未命名')}\n"
        prompt += f"- 结构构象总数: {context_data.get('total_models', 'N/A')}\n"
        prompt += f"- 最高置信度评分: {context_data.get('best_confidence', 'N/A')}\n"
        if base_conf:
            prompt += f"- 平均 pLDDT: {base_conf.get('mean_plddt', 'N/A')}\n"
            prompt += f"- pTM: {base_conf.get('ptm', 'N/A')}, ipTM: {base_conf.get('iptm', 'N/A')}\n"

        # 基础质量检查 (basic_qc)
        qc_data = analyzed_tools.get("basic_qc", {})
        if qc_data and "error" not in qc_data:
            stats = qc_data.get("stats", {})
            if stats:
                prompt += f"- pLDDT 质量分布: >90极高可信({stats.get('>90', 0)}), 70-90高可信({stats.get('70-90', 0)}), 50-70低可信({stats.get('50-70', 0)}), <50极低可信({stats.get('<50', 0)})\n"
                has_data = True
            # 兼容旧版 qc_stats 键
            if not has_data and "stats" not in qc_data:
                prompt += f"- pLDDT 质量分布: {qc_data}\n"
                has_data = True

        # 湿实验评估 (wetlab_profiler)
        wetlab = analyzed_tools.get("wetlab_profiler", {})
        if wetlab and "error" not in wetlab:
            mets = wetlab.get("metrics", {})
            if mets:
                prompt += f"- 湿实验性质: 分子量={mets.get('MW', 0):.0f}Da, 等电点pI={mets.get('pI', 0):.2f}, 不稳定指数={mets.get('Instability_Index', 0):.2f}\n"
                has_data = True

        # 口袋探测 (pocket_detect + fpocket_detect)
        pocket = analyzed_tools.get("pocket_detect", {})
        if pocket and "error" not in pocket:
            prompt += f"- 成药口袋探测: 体积={pocket.get('volume', 0):.0f}Å³, 疏水性GRAVY={pocket.get('hydrophobicity', 0):.2f}, 内衬残基数={len(pocket.get('residues', []))}\n"
            has_data = True

        fpocket = analyzed_tools.get("fpocket_detect", None)
        if fpocket and isinstance(fpocket, list) and len(fpocket) > 0:
            prompt += "- fpocket 高精度口袋:\n"
            for p in fpocket[:3]:
                prompt += f"  - 口袋体积: {p.get('Volume', 0)} Å³, Druggability Score: {p.get('Druggability Score', 0)}\n"
            has_data = True

        # Foldseek 结构检索
        foldseek = analyzed_tools.get("foldseek_search", None)
        if foldseek and isinstance(foldseek, list) and len(foldseek) > 0:
            prompt += "- Foldseek 检索匹配结构:\n"
            for m in foldseek[:5]:
                prompt += f"  - PDB ID: {m.get('PDB ID', '')}, TM-score: {m.get('TM-score', 0)}, Sequence Identity: {m.get('Sequence Identity', 0)}\n"
            has_data = True

        # PAE 分析
        pae = analyzed_tools.get("pae_analysis", {})
        if pae and "error" not in pae:
            prompt += f"- PAE 误差: 均值={pae.get('mean', 0):.2f}Å, 中位数={pae.get('median', 0):.2f}Å, 最大值={pae.get('max', 0):.2f}Å\n"
            has_data = True

        # Pfam 结构域注释
        pfam = analyzed_tools.get("pfam_annotator", {})
        if pfam and "domains" in pfam:
            domains = pfam.get("domains", [])
            if len(domains) > 0:
                prompt += f"- Pfam 结构域 ({len(domains)} 个):\n"
                for d in domains[:5]:
                    prompt += f"  - {d.get('Domain', '?')}: {d.get('Start', 0)}-{d.get('End', 0)}\n"
                has_data = True

        # 兼容旧版扁平键格式
        if 'qc_stats' in context_data:
            stats = context_data['qc_stats']
            prompt += f"- pLDDT 质量分布: >90极高可信({stats.get('>90', 0)}), 70-90高可信({stats.get('70-90', 0)}), 50-70低可信({stats.get('50-70', 0)}), <50极低可信({stats.get('<50', 0)})\n"
            has_data = True
        if 'fpocket_results' in context_data:
            pockets = context_data['fpocket_results']
            prompt += "- fpocket 探测口袋:\n"
            for p in pockets:
                prompt += f"  - 口袋体积: {p.get('Volume', 0)} Å³, Druggability Score: {p.get('Druggability Score', 0)}\n"
            has_data = True
        if 'foldseek_results' in context_data:
            matches = context_data['foldseek_results']
            prompt += "- Foldseek 检索匹配:\n"
            for m in matches:
                prompt += f"  - PDB ID: {m.get('PDB ID', '')}, TM-score: {m.get('TM-score', 0)}\n"
            has_data = True
        if 'pae_summary' in context_data:
            prompt += f"- PAE 误差: {context_data['pae_summary']}\n"
            has_data = True

        if not has_data:
            prompt += "暂无有效分析数据，请告诉用户先运行其他分析工具。\n"

        prompt += USER_PROMPT_SUFFIX
        return prompt

    def generate_stream(self, context_data):
        prompt = self.build_prompt(context_data)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                temperature=0.3
            )

            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
        except Exception as e:
            yield f"\n\n**API 请求出错**: {str(e)}\n请检查 API Key 和 Base URL 是否正确配置。"
        finally:
            del prompt
            gc.collect()