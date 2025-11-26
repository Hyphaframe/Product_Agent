"""
初始搜索节点
负责生成搜索查询并执行搜索
"""
from typing import Dict, Any
from datetime import datetime
from ..state import AgentState, SearchRecord
from langgraph.types import RunnableConfig

def initial_search(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:

    llm_client = config["configurable"]["llm_client"]

    # 获取 Tavily 搜索工具
    from ...tools.search import tavily_search

    current_idx = state["current_paragraph_index"]
    current_paragraph = state["paragraphs"][current_idx]

    # 导入提示词
    from ...prompts.prompts import SYSTEM_PROMPT_FIRST_SEARCH

    
    # 生成搜索查询

    user_content = (
        f"\n\n查询主题: {state['query']}\n"
        f"段落标题: {current_paragraph['title']}\n"
        f"段落内容: {current_paragraph['content']}"
        + SYSTEM_PROMPT_FIRST_SEARCH)
    
    messages = [
        {"role": "system", "content": "你是一个搜索查询生成专家。"},
        {"role": "user", "content": user_content}
    ]


    json_schema = {
        "type": "object",
        "properties": {
            "search_query": {"type": "string"},
            "reasoning": {"type": "string"}
        },
        "required": ["search_query", "reasoning"]
    }

    response = llm_client.chat(messages, json_schema=json_schema)
    search_query = response["search_query"]

    # 执行搜索(使用原项目的 tavily_search 函数)
    search_results = tavily_search(
        search_query,
        max_results=config["configurable"].get("max_search_results", 3),
        timeout=config["configurable"].get("search_timeout", 30),
        api_key=config["configurable"]["tavily_api_key"]
    )

    # 记录搜索历史
    search_record = SearchRecord(
        query=search_query,
        results=search_results or [],
        timestamp=datetime.now().isoformat()
    )

    # 更新段落的搜索历史
    updated_paragraphs = state["paragraphs"].copy()
    updated_paragraphs[current_idx]["search_history"].append(search_record)

    return {
        "paragraphs": updated_paragraphs
    }