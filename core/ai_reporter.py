import json
from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════
# AI 综合报告提示词模板
# 设计原则：学术严谨 + 逻辑清晰 + 多维综合 + 实用导向
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一位资深结构生物学家与计算生物学专家，擅长将 AlphaFold 3 多模态预测数据转化为专业、严谨、逻辑清晰的结构生物学综合分析报告。

【写作规范】
1. 语言风格：学术化、专业化，使用结构生物学标准术语。避免口语化表达（如"大块头""大剪刀""带教大牛"等），避免过度拟人化比喻。
2. 逻辑结构：每个结论必须有数据支撑，先陈述数据事实，再给出专业解读，最后说明生物学意义或实验建议。
3. 数据综合：不要逐条罗列数据，要将不同模块的结果交叉印证、综合分析。例如将 pLDDT 分布与 PAE 矩阵结合评估模型可靠性，将 Pfam 结构域与 Foldseek 匹配结果结合推断功能。
4. 篇幅控制：简洁精炼，每个章节聚焦核心发现，避免冗余重复。

【数据隔离红线】
严禁提及、联想或引用任何与 TCGA（癌症基因组图谱）相关的数据库内容、癌症队列、临床相关性或衍生术语。

【输出格式】
使用 Markdown 格式，包含以下章节结构（可根据实际数据可用性调整）：
- ## 一、模型质量评估（pLDDT + pTM/ipTM + PAE 综合分析）
- ## 二、蛋白理化性质与实验可行性（分子量 + pI + 不稳定指数）
- ## 三、结构域组成与功能推断（Pfam 结构域 + Foldseek 同源比对综合解读）
- ## 四、结合口袋与成药性分析（fpocket/网格法口袋探测结果）
- ## 五、综合结论与后续实验建议"""

USER_PROMPT_TEMPLATE = """请基于以下 AlphaFold 3 多模块分析数据，生成一份专业、严谨、逻辑清晰的结构生物学综合分析报告。

【数据上下文】
{context}

【报告要求】
1. 每个章节必须基于实际数据进行分析，数据缺失的模块简要说明即可，不要编造数据。
2. 在"结构域组成与功能推断"章节中，综合 Pfam 结构域注释和 Foldseek 结构相似性检索结果，推断蛋白的可能功能分类和生物学角色。
3. 在"结合口袋与成药性分析"章节中，对检测到的口袋按可成药性评分排序，评估其作为药物靶点的潜力。
4. 在"综合结论"中，用 3-5 句话概括该蛋白的结构特征、功能推测和最具价值的后续研究方向。
5. 直接输出报告正文，不要输出"好的""以下是"等开场白。"""


def generate_insight_report(context_dict: dict, api_base: str, api_key: str, model_name: str):

    # 1. 组装格式化上下文文本
    formatted_context = json.dumps(context_dict, indent=2, ensure_ascii=False)

    # 2. 填充用户提示词
    user_prompt = USER_PROMPT_TEMPLATE.format(context=formatted_context)

    # 3. 初始化标准 OpenAI 客户端，自适应兼容各类定制反代路由
    client = OpenAI(base_url=api_base, api_key=api_key)

    # 4. 流式通信链路建立
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        stream=True,
        temperature=0.3  # 低创造度，确保高结构科学严谨性
    )

    # 5. 构建支持 Streamlit 实时流式流渲染的文本生成器
    for chunk in response:
        if chunk.choices and len(chunk.choices) > 0:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                yield delta_content