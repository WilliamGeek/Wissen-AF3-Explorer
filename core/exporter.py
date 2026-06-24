class MarkdownExporter:
    """Wissen AF3 Explorer 一键导出引擎 - 负责将全景参数档案、置信度指标和 AI 洞察报告进行流水线精美排版组装"""
    
    @staticmethod
    def generate_report_markdown(context_dict: dict, ai_report_text: str) -> str:
        """
        将提取出的结构置信度特征与 LLM 专家综合洞察文本，精密组装成排版优美、层级清晰的纯 Markdown 报告文本
                
        :param context_dict: 浓缩后的置信度指标及核心任务输入参数字典 (v3.1 兼容新旧两种格式)
        :param ai_report_text: AI 智能综合洞察报告的完整文本
        :return: 经过结构化精美排版后的完整 Markdown 字符串
        """
        # v3.1 兼容：从 base_confidences 或顶层键提取置信度指标
        base_conf = context_dict.get("base_confidences", {})
        job_name = context_dict.get("project_name") or context_dict.get("job_name") or base_conf.get("job_name") or "未命名任务"
        job_id = base_conf.get("job_id") or context_dict.get("job_id") or "N/A"
        ptm = base_conf.get("ptm") or context_dict.get("ptm") or "N/A"
        iptm = base_conf.get("iptm") or context_dict.get("iptm") or "N/A"
        mean_plddt = base_conf.get("mean_plddt") or context_dict.get("mean_plddt") or 0.0
        fraction_conf = base_conf.get("fraction_confidently_predicted") or context_dict.get("fraction_confidently_predicted") or "N/A"
        model_type = base_conf.get("model_type") or context_dict.get("model_type") or "N/A"
        
        # 1. 构建排版精美的一级标题与摘要
        md_lines = []
        md_lines.append(f"# 🧬 AlphaFold 3 结构全景解析与智能综合洞察报告: {job_name}")
        md_lines.append(f"\n> 本报告由 **Wissen AF3 Explorer** 智能分析平台为您流水线生成。抹平复杂的生信文件目录，一键透视大分子核心置信度。")
        md_lines.append(f"\n---\n")
        
        # 2. 构建二级标题：AF3 参数与置信度档案（Markdown 矩阵表格）
        md_lines.append(f"## 📋 一、AlphaFold 3 参数与核心置信度档案\n")
        md_lines.append(f"以下表格汇总了该作业的基本描述特征，以及由后端智能过滤、高负荷矩阵清洗后浓缩提取的宏观评估核心指标：\n")
        
        md_lines.append(f"| 评测指标 / 任务属性 | 属性参数值 | 生物物理学评估解释 |")
        md_lines.append(f"| :--- | :--- | :--- |")
        md_lines.append(f"| **作业项目名称 (Job Name)** | `{job_name}` | 预测作业在系统中的命名标识 |")
        md_lines.append(f"| **任务唯一识别 ID (Job ID)** | `{job_id}` | 用于溯源及唯一标识该 AF3 运行任务的哈希字符串 |")
        md_lines.append(f"| **模型架构类型 (Model Type)** | `{model_type}` | 任务运行所采用的 AlphaFold 3 神经网络基础配置 |")
        
        # 格式化数值展示
        ptm_str = f"{ptm:.4f}" if isinstance(ptm, (int, float)) else str(ptm)
        iptm_str = f"{iptm:.4f}" if isinstance(iptm, (int, float)) else str(iptm)
        plddt_str = f"{mean_plddt:.4f}" if isinstance(mean_plddt, (int, float)) else str(mean_plddt)
        fraction_str = f"{fraction_conf:.4f}" if isinstance(fraction_conf, (int, float)) else str(fraction_conf)
        
        md_lines.append(f"| **全局模型分值 (TM-score / pTM)** | `{ptm_str}` | 评估全图拓扑骨架预测置信度，越接近 1.0 骨架契合度越高 |")
        md_lines.append(f"| **界面模型分值 (ipTM)** | `{iptm_str}` | 多聚体复合物专属指标，用于评估链间接触界面的预测质量 |")
        md_lines.append(f"| **平均残基置信度 (Mean pLDDT)** | `{plddt_str}` | 逐残基局部结构可信度平均分，>90 代表具备高置信度侧链精细结构 |")
        md_lines.append(f"| **高置信残基占比 (Fraction Confident)** | `{fraction_str}` | 模型整体预测中达到高可靠置信区间的残基数目所占的比例 |")
        
        md_lines.append(f"\n---\n")
        
        # 3. 构建二级标题：AI 专家综合洞察
        md_lines.append(f"## 🤖 二、AI 专家级结构综合洞察\n")
        md_lines.append(f"以下内容由系统召集的虚拟资深结构生物学专家为您流式研判，提供学术级的置信度特征解读：\n")
        md_lines.append(ai_report_text)
        
        md_lines.append(f"\n\n---\n")
        
        # 4. 构建页脚
        md_lines.append(f"**报告生成说明**：")
        md_lines.append(f"- *生成引擎*：Wissen AF3 Explorer 自动化一键导出引擎 v1.0")
        md_lines.append(f"- *设计哲学*：极简解析、科学流转、数据隔离安全护栏保障。")
        md_lines.append(f"- *免责声明*：本报告内容由 AlphaFold 3 预测指标和 AI 语言模型综合产出，仅供分子生物学及计算结构生物学科学研究参考，不作为临床医疗诊断依据。")
        
        return "\n".join(md_lines)