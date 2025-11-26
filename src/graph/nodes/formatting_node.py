"""
报告格式化节点
负责将所有段落整合为最终的 Markdown 报告
"""
from typing import Dict, Any
from ..state import AgentState
from langgraph.types import RunnableConfig

def format_report(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    
    llm_client = config["configurable"]["llm_client"]

    from ...prompts.prompts import SYSTEM_PROMPT_REPORT_FORMATTING
    from ...utils.text_processing import remove_reasoning_from_output, clean_markdown_tags

    # 准备所有段落的数据
    paragraphs_data = []
    for paragraph in state["paragraphs"]:
        paragraphs_data.append({
            "title": paragraph["title"],
            "paragraph_latest_state": paragraph["latest_summary"]
        })

        # 构建输入消息
    import json
    input_message = json.dumps(paragraphs_data, ensure_ascii=False)

    # 调用 LLM 生成格式化报告
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_REPORT_FORMATTING},
        {"role": "user", "content": input_message}
    ]

    # 不需要 JSON Schema,直接返回 Markdown 文本
    response = llm_client.chat(messages)

    # 如果 response 是字典,提取内容
    if isinstance(response, dict):
        formatted_report = response.get("content", str(response))
    else:
        formatted_report = str(response)

        # 后处理:移除推理过程和清理 Markdown 标签
    formatted_report = remove_reasoning_from_output(formatted_report)
    formatted_report = clean_markdown_tags(formatted_report)

    # 添加报告标题
    final_report = f"# {state['report_title']}\n\n{formatted_report}"

    return {
        "final_report": final_report,
        "completed": True
    }