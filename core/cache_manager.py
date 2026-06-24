"""
Wissen AF3 Explorer - 跨工具持久化缓存管理器
提供统一的缓存存储、读取、更新与清除接口，
确保工具切换后已分析数据不丢失，支持 AI 综合报告读取全量缓存。
"""
import time
import streamlit as st
from typing import Any, Dict, List, Optional


class CacheManager:
    """统一缓存管理器 —— 基于 Streamlit session_state 的持久化缓存层"""
    
    _STORE_KEY = "cache_store"
    _META_KEY = "cache_meta"
    _PARAMS_KEY = "cache_params"  # 工具参数缓存
    
    # ── 生命周期 ──────────────────────────
    
    @classmethod
    def init(cls) -> None:
        """确保缓存存储已初始化"""
        if cls._STORE_KEY not in st.session_state:
            st.session_state[cls._STORE_KEY] = {}
        if cls._META_KEY not in st.session_state:
            st.session_state[cls._META_KEY] = {}
        if cls._PARAMS_KEY not in st.session_state:
            st.session_state[cls._PARAMS_KEY] = {}
    
    @classmethod
    def reset(cls) -> None:
        """完全清空所有缓存数据"""
        st.session_state[cls._STORE_KEY] = {}
        st.session_state[cls._META_KEY] = {}
        st.session_state[cls._PARAMS_KEY] = {}
    
    # ── 写入接口 ──────────────────────────
    
    @classmethod
    def set(cls, tool_id: str, data: Any, metadata: Optional[Dict] = None) -> None:
        """
        存储工具的缓存数据
        
        :param tool_id:   工具唯一 ID (如 'basic_qc', 'pocket_detect')
        :param data:      分析结果数据 (任意可序列化对象)
        :param metadata:  可选的元信息 (如标签、描述、时间戳)
        """
        cls.init()
        st.session_state[cls._STORE_KEY][tool_id] = data
        st.session_state[cls._META_KEY][tool_id] = {
            "timestamp": time.time(),
            "label": (metadata or {}).get("label", tool_id),
            "summary": (metadata or {}).get("summary", ""),
            **(metadata or {})
        }
    
    @classmethod
    def update(cls, tool_id: str, data: Any) -> None:
        """更新已有缓存 (保留元数据不变)"""
        cls.set(tool_id, data, st.session_state.get(cls._META_KEY, {}).get(tool_id))
    
    # ── 读取接口 ──────────────────────────
    
    @classmethod
    def get(cls, tool_id: str) -> Optional[Any]:
        """
        获取单个工具的缓存数据
        
        :param tool_id: 工具唯一 ID
        :return: 缓存数据，无数据时返回 None
        """
        cls.init()
        return st.session_state[cls._STORE_KEY].get(tool_id)
    
    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """
        获取所有已分析工具的缓存数据 (供 AI 综合报告使用)
        
        :return: {tool_id: cached_data, ...}
        """
        cls.init()
        return dict(st.session_state[cls._STORE_KEY])
    
    @classmethod
    def get_meta(cls, tool_id: str) -> Dict[str, Any]:
        """获取工具的缓存元信息"""
        cls.init()
        return st.session_state[cls._META_KEY].get(tool_id, {})
    
    @classmethod
    def get_all_meta(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有工具的缓存元信息"""
        cls.init()
        return dict(st.session_state[cls._META_KEY])
    
    # ── 状态查询 ──────────────────────────
    
    @classmethod
    def is_analyzed(cls, tool_id: str) -> bool:
        """检查工具是否已有缓存数据"""
        cls.init()
        return tool_id in st.session_state[cls._STORE_KEY]
    
    @classmethod
    def get_analyzed_tools(cls) -> List[str]:
        """获取所有已分析的工具 ID 列表"""
        cls.init()
        return list(st.session_state[cls._STORE_KEY].keys())
    
    @classmethod
    def get_analyzed_count(cls) -> int:
        """获取已分析的工具数量"""
        cls.init()
        return len(st.session_state[cls._STORE_KEY])
    
    # ── 删除接口 ──────────────────────────
    
    @classmethod
    def remove(cls, tool_id: str) -> None:
        """移除指定工具的缓存"""
        cls.init()
        st.session_state[cls._STORE_KEY].pop(tool_id, None)
        st.session_state[cls._META_KEY].pop(tool_id, None)
    
    # ── 参数缓存接口 ──────────────────────
    
    @classmethod
    def save_params(cls, tool_id: str, params: Dict[str, Any]) -> None:
        """保存工具参数配置（跨工具切换持久化）"""
        cls.init()
        st.session_state[cls._PARAMS_KEY][tool_id] = params
    
    @classmethod
    def get_params(cls, tool_id: str) -> Dict[str, Any]:
        """获取已缓存的工具参数配置"""
        cls.init()
        return st.session_state[cls._PARAMS_KEY].get(tool_id, {})
    
    # ── 摘要接口 ──────────────────────────
    
    @classmethod
    def summary(cls) -> Dict[str, Any]:
        """生成缓存状态摘要"""
        cls.init()
        all_ids = cls.get_analyzed_tools()
        return {
            "total_analyzed": len(all_ids),
            "tool_ids": all_ids,
            "tools_detail": {
                tid: {
                    "label": cls.get_meta(tid).get("label", tid),
                    "timestamp": cls.get_meta(tid).get("timestamp", 0),
                    "has_data": cls.get(tid) is not None,
                }
                for tid in all_ids
            }
        }