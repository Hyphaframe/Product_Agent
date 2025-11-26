"""
总结节点
负责基于搜索结果生成段落总结
"""
from typing import Dict, Any
from ..state import AgentState
from langgraph.types import RunnableConfig


def initial_summary(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:

    llm_client = config["configurable"]["llm_client"]

    # 导入文本处理工具
    from ...utils.text_processing import format_search_results_for_prompt

    current_idx = state["current_paragraph_index"]
    current_paragraph = state["paragraphs"][current_idx]

    # 获取最新搜索结果
    if not current_paragraph["search_history"]:
        return {}  # 没有搜索结果,跳过

    latest_search = current_paragraph["search_history"][-1]

    # 格式化搜索结果
    formatted_results = format_search_results_for_prompt(
        latest_search["results"],
        max_length=config["configurable"].get("max_content_length", 20000)
    )

    # 导入提示词
    from ...prompts.prompts import SYSTEM_PROMPT_FIRST_SUMMARY

    user_content = (
        f"\n\n查询主题: {state['query']};\n"
        f"段落标题: {current_paragraph['title']};\n"
        f"段落内容: {current_paragraph['content']};\n"
        f"搜索查询: {latest_search['query']};\n"
        f"搜索结果: {formatted_results}"
    + SYSTEM_PROMPT_FIRST_SUMMARY)
    # 生成总结
    messages = [
        {"role": "system", "content": "你是一个专业的内容总结专家。"},
        {"role": "user", "content": user_content}
    ]



    json_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"}
        },
        "required": ["summary"]
    }

    response = llm_client.chat(messages, json_schema=json_schema)
    summary = response["summary"]

    # 更新段落内容
    updated_paragraphs = state["paragraphs"].copy()
    updated_paragraphs[current_idx]["content"] = summary
    updated_paragraphs[current_idx]["latest_summary"] = summary

    return {
        "paragraphs": updated_paragraphs
    }