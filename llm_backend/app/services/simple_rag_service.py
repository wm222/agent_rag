from pathlib import Path
from typing import Dict, List, Any
import json
import re
import time
import uuid

import aiohttp
import faiss
import numpy as np
import PyPDF2
import shutil

from app.core.config import settings


class SimpleRAGService:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.chat_model = settings.OLLAMA_CHAT_MODEL
        self.embedding_model = settings.OLLAMA_EMBEDDING_MODEL
        self.index_root = Path("rag_indexes")
        self.upload_root = Path("rag_uploads")
        self.index_root.mkdir(parents=True, exist_ok=True)
        self.upload_root.mkdir(parents=True, exist_ok=True)

    def _safe_filename(self, filename: str) -> str:
        name = Path(filename).name
        return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fa5._-]+", "_", name)

    async def save_upload_file(self, file, user_id: int) -> Path:
        user_dir = self.upload_root / f"user_{user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{int(time.time())}_{self._safe_filename(file.filename)}"
        file_path = user_dir / filename

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        return file_path

    def _read_file(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            texts = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        texts.append(f"[第{i + 1}页]\n{page_text}")
            return "\n\n".join(texts)

        if suffix in {".txt", ".md", ".csv"}:
            return file_path.read_text(encoding="utf-8", errors="ignore")

        raise ValueError(f"暂不支持的文件类型: {suffix}，建议先测试 PDF/TXT/MD 文件")

    def _split_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end == len(text):
                break

            start = max(0, end - overlap)

        return chunks

    async def _embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("没有可向量化的文本")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.embedding_model,
                    "input": texts
                },
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status != 200:
                    detail = await response.text()
                    raise RuntimeError(f"Ollama embedding 调用失败: {response.status}, {detail}")

                data = await response.json()
                embeddings = data.get("embeddings")

                if not embeddings:
                    raise RuntimeError(f"Ollama 没有返回 embeddings: {data}")

        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        return vectors

    async def create_index(self, file_path: Path, original_name: str, user_id: int) -> Dict[str, Any]:
        text = self._read_file(file_path)
        chunks = self._split_text(text)

        if not chunks:
            raise ValueError("文档没有提取到有效文本，可能是扫描版 PDF 或空文件")

        vectors = await self._embed(chunks)

        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)

        index_id = uuid.uuid4().hex
        index_dir = self.index_root / f"user_{user_id}" / index_id
        index_dir.mkdir(parents=True, exist_ok=True)

        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_docs.append({
                "chunk_id": i,
                "text": chunk,
                "metadata": {
                    "source": original_name,
                    "saved_path": str(file_path)
                }
            })

        faiss.write_index(index, str(index_dir / "index.faiss"))

        with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunk_docs, f, ensure_ascii=False, indent=2)

        with open(index_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "index_id": index_id,
                "user_id": user_id,
                "original_name": original_name,
                "saved_path": str(file_path),
                "embedding_model": self.embedding_model,
                "chat_model": self.chat_model,
                "chunk_count": len(chunks),
                "dimension": dimension,
                "created_at": int(time.time())
            }, f, ensure_ascii=False, indent=2)

        return {
            "status": "success",
            "index_id": index_id,
            "user_id": user_id,
            "filename": original_name,
            "chunk_count": len(chunks),
            "dimension": dimension
        }

    def _load_index(self, index_id: str, user_id: int):
        index_dir = self.index_root / f"user_{user_id}" / index_id
        index_path = index_dir / "index.faiss"
        chunks_path = index_dir / "chunks.json"
        meta_path = index_dir / "meta.json"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"找不到索引: user_id={user_id}, index_id={index_id}")

        index = faiss.read_index(str(index_path))

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        return index, chunks, meta

    def list_indexes(self, user_id: int) -> List[Dict[str, Any]]:
        """
        列出当前用户已经创建的所有 RAG 索引
        """
        user_index_dir = self.index_root / f"user_{user_id}"

        if not user_index_dir.exists():
            return []

        results = []

        for index_dir in user_index_dir.iterdir():
            if not index_dir.is_dir():
                continue

            meta_path = index_dir / "meta.json"

            if not meta_path.exists():
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                results.append({
                    "index_id": meta.get("index_id", index_dir.name),
                    "user_id": meta.get("user_id", user_id),
                    "original_name": meta.get("original_name", ""),
                    "saved_path": meta.get("saved_path", ""),
                    "embedding_model": meta.get("embedding_model", ""),
                    "chat_model": meta.get("chat_model", ""),
                    "chunk_count": meta.get("chunk_count", 0),
                    "dimension": meta.get("dimension", 0),
                    "created_at": meta.get("created_at", 0),
                })

            except Exception:
                continue

        results.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return results

    def delete_index(self, index_id: str, user_id: int) -> Dict[str, Any]:
        """
        删除当前用户的某个 RAG 索引
        """
        index_dir = self.index_root / f"user_{user_id}" / index_id

        if not index_dir.exists():
            raise FileNotFoundError(f"找不到索引: user_id={user_id}, index_id={index_id}")

        shutil.rmtree(index_dir)

        return {
            "status": "success",
            "message": "索引删除成功",
            "user_id": user_id,
            "index_id": index_id
        }

    async def retrieve(self, question: str, index_id: str, user_id: int, top_k: int = 4):
        index, chunks, meta = self._load_index(index_id, user_id)

        query_vector = await self._embed([question])
        top_k = max(1, min(top_k, len(chunks)))

        scores, ids = index.search(query_vector, top_k)

        contexts = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue

            item = chunks[int(idx)]
            contexts.append({
                "score": float(score),
                "chunk_id": item["chunk_id"],
                "content": item["text"],
                "metadata": item["metadata"]
            })

        return {
            "contexts": contexts,
            "meta": meta
        }

    async def answer(self, question: str, index_id: str, user_id: int, top_k: int = 4):
        retrieved = await self.retrieve(question, index_id, user_id, top_k)
        contexts = retrieved["contexts"]

        context_text = "\n\n".join([
            f"[片段{i + 1} | score={c['score']:.4f}]\n{c['content']}"
            for i, c in enumerate(contexts)
        ])

        messages = [
            {
                "role": "system",
                "content": "你是一个严谨的文档问答助手。请优先依据给定文档片段回答；如果文档片段中没有答案，必须明确说明文档中没有找到依据，不要编造。"
            },
            {
                "role": "user",
                "content": f"已检索到的文档片段如下：\n\n{context_text}\n\n用户问题：{question}\n\n请用中文回答，并在必要时指出依据来自哪些片段。"
            }
        ]

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.chat_model,
                    "messages": messages,
                    "stream": False
                },
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status != 200:
                    detail = await response.text()
                    raise RuntimeError(f"Ollama chat 调用失败: {response.status}, {detail}")

                data = await response.json()

        return {
            "answer": data.get("message", {}).get("content", ""),
            "contexts": contexts,
            "index_id": index_id,
            "user_id": user_id,
            "meta": retrieved["meta"]
        }