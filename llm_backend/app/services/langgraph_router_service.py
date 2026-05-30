import os
import inspect
from typing import TypedDict, List, Dict, Optional, Any

import aiohttp
from langgraph.graph import StateGraph, END

from app.services.simple_rag_service import SimpleRAGService


class AgentState(TypedDict, total=False):
    user_id: int
    conversation_id: Optional[int]
    question: str
    messages: List[Dict[str, str]]

    rag_enabled: bool
    rag_index_id: Optional[str]
    top_k: int

    force_route: Optional[str]
    route: str
    answer: str
    contexts: List[Dict[str, Any]]
    tool_result: Any


class LangGraphRouterService:
    """
    轻量级 LangGraph Agent 路由服务。

    当前支持三个分支：
    1. chat：普通聊天
    2. rag：知识库问答
    3. tool：工具调用，目前先实现查询知识库索引列表
    """

    def __init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")

        workflow = StateGraph(AgentState)

        workflow.add_node("router", self.router_node)
        workflow.add_node("chat", self.chat_node)
        workflow.add_node("rag", self.rag_node)
        workflow.add_node("tool", self.tool_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            self.choose_next_node,
            {
                "chat": "chat",
                "rag": "rag",
                "tool": "tool",
            }
        )

        workflow.add_edge("chat", END)
        workflow.add_edge("rag", END)
        workflow.add_edge("tool", END)

        self.graph = workflow.compile()

    async def run(
        self,
        user_id: int,
        question: str,
        messages: Optional[List[Dict[str, str]]] = None,
        conversation_id: Optional[int] = None,
        rag_enabled: bool = False,
        rag_index_id: Optional[str] = None,
        top_k: int = 4,
        force_route: Optional[str] = None,
    ) -> Dict[str, Any]:
        state: AgentState = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "question": question,
            "messages": messages or [{"role": "user", "content": question}],
            "rag_enabled": rag_enabled,
            "rag_index_id": rag_index_id,
            "top_k": top_k,
            "force_route": force_route,
            "route": "",
            "answer": "",
            "contexts": [],
        }

        result = await self.graph.ainvoke(state)

        return {
            "status": "success",
            "route": result.get("route", ""),
            "answer": result.get("answer", ""),
            "contexts": result.get("contexts", []),
            "tool_result": result.get("tool_result"),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "rag_index_id": rag_index_id,
        }

    async def router_node(self, state: AgentState) -> Dict[str, Any]:
        """
        路由节点：判断问题应该走普通聊天、RAG，还是工具调用。

        第一版先用规则路由，稳定、容易解释。
        后续可以升级为 LLM Router。
        """
        question = state.get("question", "") or ""
        force_route = state.get("force_route")

        if force_route in {"chat", "rag", "tool"}:
            return {"route": force_route}

        q = question.lower()

        tool_keywords = [
            "有哪些知识库",
            "知识库列表",
            "索引列表",
            "有哪些索引",
            "上传了哪些文档",
            "已有文档",
            "已有知识库",
            "list indexes",
            "show indexes",
        ]

        rag_keywords = [
            "文档",
            "文件",
            "资料",
            "论文",
            "pdf",
            "docx",
            "知识库",
            "根据上传",
            "根据这篇",
            "根据这份",
            "这篇论文",
            "这份文档",
            "这篇文档",
            "材料中",
            "文中",
        ]

        if any(k in q for k in tool_keywords):
            return {"route": "tool"}

        # 如果用户显式开启了 RAG 并且已有索引，则优先走 RAG
        if state.get("rag_enabled") and state.get("rag_index_id"):
            return {"route": "rag"}

        # 如果问题明显是在问文档，且有索引，也走 RAG
        if any(k in q for k in rag_keywords) and state.get("rag_index_id"):
            return {"route": "rag"}

        return {"route": "chat"}

    def choose_next_node(self, state: AgentState) -> str:
        route = state.get("route", "chat")

        if route not in {"chat", "rag", "tool"}:
            return "chat"

        return route

    async def chat_node(self, state: AgentState) -> Dict[str, Any]:
        """
        普通聊天节点：直接调用 Ollama chat。
        """
        messages = state.get("messages") or [
            {"role": "user", "content": state.get("question", "")}
        ]

        # 加一个轻量 system prompt，避免模型忘记身份
        final_messages = [
            {
                "role": "system",
                "content": "你是一个中文智能助手，请根据用户问题进行简洁、准确的回答。"
            }
        ] + messages

        payload = {
            "model": self.chat_model,
            "messages": final_messages,
            "stream": False,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.ollama_base_url}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return {
                        "route": "chat",
                        "answer": f"普通聊天模型调用失败：{text}",
                    }

                data = await resp.json()

        answer = data.get("message", {}).get("content", "")

        return {
            "route": "chat",
            "answer": answer,
        }

    async def rag_node(self, state: AgentState) -> Dict[str, Any]:
        """
        RAG 节点：调用已有 SimpleRAGService。
        """
        rag_index_id = state.get("rag_index_id")

        if not rag_index_id:
            return {
                "route": "rag",
                "answer": "已识别为知识库问题，但当前没有选择知识库索引。请先上传文档或选择已有知识库。",
                "contexts": [],
            }

        rag_service = SimpleRAGService()

        result = await rag_service.answer(
            question=state.get("question", ""),
            index_id=rag_index_id,
            user_id=state.get("user_id"),
            top_k=state.get("top_k", 4),
        )

        return {
            "route": "rag",
            "answer": result.get("answer", ""),
            "contexts": result.get("contexts", []),
        }

    async def tool_node(self, state: AgentState) -> Dict[str, Any]:
        """
        工具节点：当前先实现查询当前用户已有知识库索引。
        """
        rag_service = SimpleRAGService()
        user_id = state.get("user_id")

        result = rag_service.list_indexes(user_id)

        if inspect.isawaitable(result):
            result = await result

        indexes = result.get("indexes", []) if isinstance(result, dict) else []

        if not indexes:
            answer = "当前用户还没有可用的知识库索引。可以先上传文档建立索引。"
        else:
            lines = ["当前用户已有知识库索引："]
            for i, item in enumerate(indexes, start=1):
                name = item.get("original_name") or item.get("filename") or "未命名文档"
                index_id = item.get("index_id", "")
                chunk_count = item.get("chunk_count", 0)
                lines.append(f"{i}. {name}，index_id={index_id}，chunks={chunk_count}")
            answer = "\n".join(lines)

        return {
            "route": "tool",
            "answer": answer,
            "tool_result": result,
        }
