"""
Deep Search Agent主类 - LangGraph版本
整合LangGraph图结构,实现完整的深度搜索流程
"""

import json
import os
from datetime import datetime
import time
from typing import Optional, Dict, Any

from .llms import OpenAILLM, BaseLLM
from .graph import create_research_graph, AgentState
from .utils import Config, load_config


class DeepSearchAgent:
    """Deep Search Agent主类 - 使用LangGraph实现"""

    def __init__(self, config: Optional[Config] = None):
        """
        初始化Deep Search Agent

        Args:
            config: 配置对象,如果不提供则自动加载
        """
        # 加载配置
        self.config = config or load_config()

        # 初始化LLM客户端
        self.llm_client = self._initialize_llm()

        # 创建LangGraph图
        self.graph = create_research_graph()

        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)

        print(f"Deep Search Agent 已初始化 (LangGraph版本)")
        print(f"使用LLM: {self.llm_client.get_model_info()}")

    def _initialize_llm(self) -> BaseLLM:
        """初始化LLM客户端"""
        
        from .llms.openai_llm import OpenAILLM  
      
        # 使用 OpenAILLM 客户端,设置 base_url 指向硅基流动  
        return OpenAILLM(  
            api_key=self.config.openai_api_key,  # 使用您的硅基流动 API Key  
            model_name=self.config.openai_model,  # 使用硅基流动支持的模型名称  
            base_url="https://api.siliconflow.cn/v1"  # 硅基流动的 API 端点  
        )


    from typing import Generator, Dict, Any, Optional   # 引入生成器类型提示
    import time

    def research(
        self,
        query: str,
        save_report: bool = True,
        *,
        stream_config: Optional[Dict[str, Any]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        执行深度研究，以生成器方式实时返回节点进度与最终报告。

        Args:
            query: 研究问题
            save_report: 是否保存报告
            stream_config: 透传给 graph.stream 的额外配置（如 debug、recursion_limit）

        Yields:
            {"node": 节点名, "state": 当前状态快照}
            最后一条为 {"node": "completed", "report": 最终报告}
        """
        start_time = time.time()
        print(f"\n{'='*60}\n开始深度研究: {query}\n{'='*60}")

        try:
            # 1. 初始状态
            initial_state: AgentState = {
                "query": query,
                "report_title": "",
                "paragraphs": [],
                "current_paragraph_index": 0,
                "reflection_count": 0,
                "max_reflections": self.config.max_reflections,
                "final_report": None,
                "completed": False,
            }

            # 2. 默认配置 & 支持外部透传
            config = {
                "configurable": {
                    "llm_client": self.llm_client,
                    "tavily_api_key": self.config.tavily_api_key,
                    "max_search_results": self.config.max_search_results,
                    "search_timeout": self.config.search_timeout,
                    "max_content_length": self.config.max_content_length,
                    "max_reflections": self.config.max_reflections,
                },
                "recursion_limit": 100,          # 防死循环兜底
                "debug": False,                  # 默认关闭调试日志
            }
            if stream_config:
                config.update(stream_config)

            # 3. 流式执行
            print("\n执行研究工作流...")
            final_state = None
            for chunk in self.graph.stream(initial_state, config):
                node_name = next(iter(chunk))   # 更安全地取键
                node_output = chunk[node_name]
                final_state = node_output

                yield {"node": node_name, "state": node_output}

            # 4. 后处理
            if not final_state:
                raise RuntimeError("工作流未产生任何状态")

            final_report = final_state.get("final_report")
            if not final_report:
                raise RuntimeError("最终报告为空，可能图未正确填充 final_report 字段")

            if save_report:
                self._save_report(final_report, query)

            end_time = time.time()
            run_time = end_time - start_time
            print("\n深度研究完成！")
            print(f"总用时: {run_time:.2f} 秒")
            yield {"node": "completed", "report": final_report, "run_time": run_time}

        except Exception as e:
            print(f"[research] 研究过程中发生错误: {e}")
            raise

        

    def _save_report(self, report_content: str, query: str):
        """保存报告到文件"""
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
        query_safe = query_safe.replace(' ', '_')[:30]

        filename = f"deep_search_report_{query_safe}_{timestamp}.md"
        filepath = os.path.join(self.config.output_dir, filename)

        # 保存报告
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"报告已保存到: {filepath}")

    def get_progress_summary(self) -> Dict[str, Any]:
        """获取进度摘要 - LangGraph版本暂不支持"""
        return {
            "message": "LangGraph版本使用内置检查点机制,请使用LangGraph的状态查询API"
        }


def create_agent(config_file: Optional[str] = None) -> DeepSearchAgent:
    """
    创建Deep Search Agent实例的便捷函数

    Args:
        config_file: 配置文件路径

    Returns:
        DeepSearchAgent实例
    """
    config = load_config(config_file)
    return DeepSearchAgent(config)