"""
LangGraph 图构建器
定义研究工作流的状态图结构
"""
from typing import Any, Dict, Literal
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    generate_structure,
    initial_search,
    initial_summary,
    reflection_search,
    reflection_summary,
    format_report
)


def should_reflect(state: AgentState) -> Literal["reflect", "next_paragraph", "format"]:

    current_idx = state["current_paragraph_index"]
    current_paragraph = state["paragraphs"][current_idx]

    # 检查是否达到最大反思次数
    if current_paragraph["reflection_count"] < state["max_reflections"]:
        return "reflect"

        # 标记当前段落完成
    # state["paragraphs"][current_idx]["completed"] = True

    # 检查是否还有未完成的段落
    if current_idx < len(state["paragraphs"]) - 1:
        return "next_paragraph"

        # 所有段落完成
    return "format"


def check_reflection_complete(state: AgentState) -> Literal["continue", "done"]:
    """
    检查反思是否完成

    Returns:
        - "continue": 继续反思搜索
        - "done": 反思完成,返回总结节点
    """
    # current_idx = state["current_paragraph_index"]
    # current_paragraph = state["paragraphs"][current_idx]

    # 简化逻辑:每次反思后都返回总结节点
    # 实际可以根据搜索结果质量判断
    return "done"


def move_to_next_paragraph(state: AgentState) -> AgentState:
    """移动到下一段落"""
    state["current_paragraph_index"] += 1
    state["reflection_count"] = 0
    return state








def create_research_graph(config=None):
    """
    创建研究工作流的 StateGraph

    Args:
        config: 配置对象,包含 llm_client, search_tool, max_reflections 等

    Returns:
        编译后的 LangGraph 图对象
    """
    # 创建状态图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("structure", generate_structure)
    workflow.add_node("search", initial_search)
    workflow.add_node("summary", initial_summary)
    workflow.add_node("reflect", reflection_search)
    workflow.add_node("reflect_summary", reflection_summary)
    workflow.add_node("next_paragraph", move_to_next_paragraph)
    workflow.add_node("format", format_report)

    # 设置入口点
    workflow.set_entry_point("structure")

    # 定义边
    workflow.add_edge("structure", "search")
    workflow.add_edge("search", "summary")

    # 条件边:从 summary 决定下一步
    workflow.add_conditional_edges(
        "summary",
        should_reflect,
        {
            "reflect": "reflect",
            "next_paragraph": "next_paragraph",
            "format": "format"
        }
    )

    # 反思循环
    workflow.add_edge("reflect", "reflect_summary")
    workflow.add_edge("reflect_summary", "summary") 
    # workflow.add_conditional_edges(
    #     "reflect_summary",
    #     check_reflection_complete,
    #     {
    #         "continue": "reflect",
    #         "done": "summary"
    #     }
    # )

    # 下一段落回到搜索
    workflow.add_edge("next_paragraph", "search")

    # 格式化后结束
    workflow.add_edge("format", END)

    # 编译图
    return workflow.compile()