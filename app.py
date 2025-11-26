

"""
Streamlit Web界面 - LangGraph版本
自动读取配置文件中的API密钥，提供友好的Web界面进行深度搜索；
支持实时进度显示和结果分标签页展示。
"""
import os
import sys

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import streamlit as st
from src import DeepSearchAgent, Config
from src.utils.config import load_config


def main():
    # -------------------- 页面配置 --------------------
    st.set_page_config(
        page_title="Deep Search Agent (LangGraph版本)",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Deep Search Agent (LangGraph版本)")
    st.markdown("基于LangGraph的深度搜索AI代理 - 自动读取配置文件中的API密钥")

    # -------------------- 侧边栏配置 --------------------
    try:
        default_config = load_config()
        has_config_file = True
        st.sidebar.success("✅ 已检测到配置文件，API Key 已自动填充")
    except Exception:
        default_config = None
        has_config_file = False
        st.sidebar.warning("⚠️ 未找到配置文件，请手动输入API密钥")

    with st.sidebar:
        st.header("⚙️ 配置")

        # --- API 密钥 ---
        st.subheader("API密钥")
        openai_api_key = st.text_input(
            "OpenAI/硅基流动 API Key",
            value=default_config.openai_api_key if has_config_file else "",
            type="password",
            help="从配置文件自动读取，或手动输入",
        )
        openai_model = st.text_input(
            "模型名称",
            value=default_config.openai_model if has_config_file else "deepseek-ai/DeepSeek-V3",
            help="例如：deepseek-ai/DeepSeek-V3 (硅基流动) 或 gpt-4o-mini (OpenAI)",
        )
        tavily_api_key = st.text_input(
            "Tavily API Key",
            value=default_config.tavily_api_key if has_config_file else "",
            type="password",
            help="从配置文件自动读取，或手动输入",
        )

        # --- 研究参数 ---
        st.subheader("研究参数")
        max_reflections = st.slider(
            "反思次数",
            min_value=0,
            max_value=5,
            value=default_config.max_reflections if has_config_file else 2,
            help="每个段落的反思搜索次数",
        )
        max_search_results = st.slider(
            "搜索结果数",
            min_value=1,
            max_value=10,
            value=default_config.max_search_results if has_config_file else 3,
            help="每次搜索返回的结果数量",
        )
        max_content_length = st.number_input(
            "内容最大长度",
            min_value=5000,
            max_value=50000,
            value=default_config.max_content_length if has_config_file else 20000,
            step=5000,
            help="搜索内容的最大字符数",
        )
        output_dir = st.text_input(
            "报告保存目录",
            value=default_config.output_dir if has_config_file else "reports",
            help="报告文件的保存位置",
        )

        st.markdown("---")
        st.markdown("### 关于")
        st.markdown(
            """
            这是 Deep Search Agent 的 LangGraph 版本，使用声明式图结构实现研究工作流。

            **新特性：**
            - 实时进度显示
            - 可视化工作流阶段
            - 结果分标签页展示
            - 自动读取配置文件
            """
        )

    # -------------------- 主界面输入 --------------------
    st.header("📝 研究查询")
    query = st.text_area(
        "输入您的研究问题",
        height=100,
        placeholder="例如：2025年人工智能发展趋势",
        help="输入您想要深度研究的问题",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        start_research = st.button("🚀 开始研究", type="primary", use_container_width=True)
    with col2:
        save_report = st.checkbox("保存报告到文件", value=True)

    # -------------------- 研究执行 --------------------
    if start_research:
        # 简单校验
        if not tavily_api_key:
            st.error("❌ 请输入 Tavily API Key")
            return
        if not openai_api_key:
            st.error("❌ 请输入 OpenAI/硅基流动 API Key")
            return
        if not query.strip():
            st.error("❌ 请输入研究问题")
            return

        try:
            # 构造配置
            config = Config(
                openai_api_key=openai_api_key,
                tavily_api_key=tavily_api_key,
                default_llm_provider="openai",
                openai_model=openai_model,
                max_reflections=max_reflections,
                max_search_results=max_search_results,
                max_content_length=max_content_length,
                output_dir=output_dir,
                save_intermediate_states=False,
            )

            # 初始化 Agent
            with st.spinner("正在初始化 Deep Search Agent (LangGraph版本)..."):
                agent = DeepSearchAgent(config)
            st.success("✅ Agent 初始化成功")

            # ---- 实时进度展示 ----
            st.markdown("---")
            st.header("🔄 研究进度")

            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            # 节点中文映射
            node_names = {
                "structure": "📋 生成报告结构",
                "search": "🔍 执行搜索",
                "summary": "📝 生成总结",
                "reflect": "🤔 反思搜索",
                "reflect_summary": "✍️ 更新总结",
                "next_paragraph": "➡️ 移动到下一段落",
                "format": "📄 格式化最终报告",
            }

            final_report = None
            for progress_data in agent.research(query, save_report=save_report):
                if progress_data["node"] == "completed":
                    final_report = progress_data["report"]
                    status_placeholder.success("✅ 研究完成！")
                    break
                else:
                    node = progress_data["node"]
                    state = progress_data["state"]
                    node_display = node_names.get(node, node)
                    status_placeholder.info(f"当前阶段：{node_display}")

                    # 段落进度条
                    if "current_paragraph_index" in state and "paragraphs" in state:
                        current_idx = state["current_paragraph_index"]
                        total = len(state["paragraphs"])
                        if total > 0:
                            progress_placeholder.progress(
                                (current_idx + 1) / total,
                                text=f"段落进度：{current_idx + 1}/{total}",
                            )

            # -------------------- 结果展示 --------------------
            if final_report:
                st.markdown("---")
                st.header("📊 研究结果")
                tab1, tab2 = st.tabs(["📄 最终报告", "💾 下载"])
                with tab1:
                    st.subheader("⏱️ 运行统计")  
                    st.metric("运行时间", f"{progress_data['run_time']:.2f} 秒")
                    st.markdown(final_report)
                with tab2:
                    st.download_button(
                        label="📥 下载 Markdown 报告",
                        data=final_report,
                        file_name=f"deep_search_report_{query[:20]}.md",
                        mime="text/markdown",
                    )

        except Exception as e:
            st.error(f"❌ 研究过程中发生错误：{str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()