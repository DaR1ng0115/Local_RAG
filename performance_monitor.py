import time 
from typing import Any,Dict,List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class PerformanceMonitorHandler(BaseCallbackHandler):
    """用于监测和记录RAG系统性能的回调处理器"""

    def __init__(self):
        self.retrieval_start_time = None
        self.llm_start_time = None
        self.token_time = []

        self.metrics = {
            "retrieval_latency": 0.0,
            "ttft": 0,   # Time To First Token
            "tps": 0,    # Tokens Per Second
            "total_tokens": 0,
            "total_duration": 0
        }
        self.start_time = None

#----------------监视检索器----------------
    def on_retriever_start(self, serialized:Dict[str, Any], query:str, **kwargs: Any) -> Any:
        self.retrieval_start_time = time.perf_counter()

    def on_retriever_end(self, documents, **kwargs: Any) -> Any:
        if self.retrieval_start_time:
            self.metrics["retrieval_latency"] = time.perf_counter() - self.retrieval_start_time
    
#----------------监控LLM流式生成----------------
    def on_llm_start(self, serialized:Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        self.llm_start_time = time.perf_counter()
        self.token_time = []
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.token_time:
            self.token_time.append(time.perf_counter())
        else:
            self.token_time.append(time.perf_counter())
        
        print(".", end="", flush=True)
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        if not self.token_time:
            return
        
        self.metrics["ttft"] = self.token_time[0] - self.llm_start_time

        if response.llm_output and 'token_usage' in response.llm_output:
            self.metrics["total_tokens"] = response.llm_output['token_usage'].get('total_tokens', 0)
        else:
            self.metrics["total_tokens"] = len(self.token_time)
        
        generation_duration = self.token_time[-1] - self.token_time[0]

        if generation_duration > 0:
            self.metrics["tps"] = self.metrics["total_tokens"] / generation_duration        