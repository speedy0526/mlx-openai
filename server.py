"""
OpenAI-compatible MLX inference server
提供与 OpenAI API 兼容的聊天完成接口
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union, AsyncGenerator, Literal
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import re
import time
import uuid
from typing_extensions import Annotated

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 全局模型和tokenizer
model = None
tokenizer = None
model_path = "./models/qwen3-8b"


# ==================== 数据模型 ====================


class ChatMessage(BaseModel):
    """聊天消息"""

    role: Literal["system", "user", "assistant"] = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completions API 请求"""

    model: str = Field(default="qwen3-8b", description="模型名称")
    messages: List[ChatMessage] = Field(..., description="消息列表")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="采样温度")
    max_tokens: int = Field(default=8192, ge=1, description="最大生成token数")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="nucleus sampling参数")
    top_k: Optional[int] = Field(default=None, ge=0, description="top_k采样")
    frequency_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="频率惩罚"
    )
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="存在惩罚")
    stream: bool = Field(default=False, description="是否流式输出")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="停止序列")


class Usage(BaseModel):
    """Token使用情况"""

    prompt_tokens: int = Field(default=0, description="输入token数")
    completion_tokens: int = Field(default=0, description="输出token数")
    total_tokens: int = Field(default=0, description="总token数")
    completion_tokens_details: Optional[Dict[str, Any]] = Field(
        default=None, description="输出token详情（包含推理token等）"
    )


class ChatCompletionChoice(BaseModel):
    """聊天完成选择"""

    index: int = Field(..., description="选择索引")
    message: ChatMessage = Field(..., description="消息")
    finish_reason: str = Field(default="stop", description="结束原因")


class ChatCompletionResponse(BaseModel):
    """OpenAI Chat Completions API 响应"""

    id: str = Field(..., description="响应ID")
    object: Literal["chat.completion"] = Field(
        default="chat.completion", description="对象类型"
    )
    created: int = Field(..., description="创建时间戳")
    model: str = Field(..., description="模型名称")
    choices: List[ChatCompletionChoice] = Field(..., description="选择列表")
    usage: Usage = Field(..., description="token使用情况")


class Delta(BaseModel):
    """流式增量"""

    role: Optional[str] = Field(default=None, description="角色")
    content: Optional[str] = Field(default=None, description="内容")
    reasoning_content: Optional[str] = Field(
        default=None, description="推理过程（兼容OpenAI reasoning API）"
    )


class ChatCompletionChunk(BaseModel):
    """流式响应块"""

    id: str = Field(..., description="响应ID")
    object: Literal["chat.completion.chunk"] = Field(
        default="chat.completion.chunk", description="对象类型"
    )
    created: int = Field(..., description="创建时间戳")
    model: str = Field(..., description="模型名称")
    choices: List[Dict[str, Any]] = Field(..., description="选择列表")


class ModelInfo(BaseModel):
    """模型信息"""

    id: str = Field(..., description="模型ID")
    object: Literal["model"] = Field(default="model", description="对象类型")
    created: int = Field(..., description="创建时间戳")
    owned_by: str = Field(default="mlx", description="所有者")


class ModelsResponse(BaseModel):
    """模型列表响应"""

    object: Literal["list"] = Field(default="list", description="对象类型")
    data: List[ModelInfo] = Field(..., description="模型列表")


# ==================== 服务生命周期 ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭时的生命周期管理"""
    global model, tokenizer

    # 启动时加载模型
    logger.info(f"Loading model from {model_path}...")
    try:
        model, tokenizer = load(model_path)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    # 关闭时清理资源
    logger.info("Shutting down server...")


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="MLX OpenAI-Compatible API",
    description="OpenAI-compatible API for MLX inference",
    version="0.1.0",
    lifespan=lifespan,
)


# ==================== 辅助函数 ====================


def clean_response_with_thinking(text: str) -> Dict[str, Any]:
    """
    清理响应文本，提取 think 标签内容和最终答案
    兼容 OpenAI reasoning API 的处理方式

    Args:
        text: 原始响应文本

    Returns:
        dict: 包含 cleaned_content（清理后的内容）、
              reasoning_content（推理过程，如果有的话）和
              reasoning_tokens（推理token数估算）
    """
    result = {
        "cleaned_content": text,
        "reasoning_content": None,
        "reasoning_tokens": 0,
        "has_reasoning": False,
    }

    # 处理 think 标签
    start_pattern = r"<\|thinking\|>"
    end_pattern = r"<\|/thinking\|>"

    # 查找开始和结束标签
    start_match = re.search(start_pattern, text)
    end_match = re.search(end_pattern, text)

    if start_match and end_match:
        start_pos = start_match.end()
        end_pos = end_match.start()

        if start_pos < end_pos:
            result["reasoning_content"] = text[start_pos:end_pos].strip()
            result["has_reasoning"] = True
            # 移除 think 标签及其内容
            result["cleaned_content"] = (
                text[: start_match.start()] + text[end_match.end() :]
            ).strip()
            # 估算推理 token 数
            result["reasoning_tokens"] = max(1, len(result["reasoning_content"]) // 4)

    # 清理多余空行
    result["cleaned_content"] = re.sub(r"\n{3,}", "\n\n", result["cleaned_content"]).strip()

    return result


def format_messages(messages: List[ChatMessage]) -> str:
    """将消息列表格式化为prompt字符串"""
    chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

    # 使用tokenizer的chat_template
    if tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            chat_messages, add_generation_prompt=True, tokenize=False
        )
        return prompt
    else:
        # 回退到简单格式
        prompt = ""
        for msg in chat_messages:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
            elif msg["role"] == "system":
                prompt += f"System: {msg['content']}\n"
        prompt += "Assistant:"
        return prompt


async def openai_stream_generator(
    response: str, request_id: str, created: int, model_name: str
) -> AsyncGenerator[str, None]:
    """OpenAI流式响应生成器"""
    cleaned = clean_response_with_thinking(response)

    # 如果有推理内容，先发送推理内容
    if cleaned["has_reasoning"]:
        reasoning_content = cleaned["reasoning_content"]
        for char in reasoning_content:
            chunk = ChatCompletionChunk(
                id=request_id,
                created=created,
                model=model_name,
                choices=[
                    {
                        "index": 0,
                        "delta": {"reasoning_content": char},
                        "finish_reason": None,
                    }
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            await asyncio.sleep(0.005)

    # 发送主要内容的第一个chunk（包含role）
    first_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model_name,
        choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # 逐字符发送内容
    content = cleaned["cleaned_content"]
    for char in content:
        chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model_name,
            choices=[
                {"index": 0, "delta": {"content": char}, "finish_reason": None}
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        await asyncio.sleep(0.01)

    # 发送结束chunk
    final_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model_name,
        choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ==================== API 端点 ====================


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """列出可用模型（OpenAI风格）"""
    return ModelsResponse(
        data=[
            ModelInfo(id="qwen3-8b", created=int(time.time()), owned_by="mlx"),
            ModelInfo(id="yuntu-llm-2b", created=int(time.time()), owned_by="mlx"),
        ]
    )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """获取指定模型信息"""
    available_models = ["qwen3-8b", "yuntu-llm-2b"]

    if model_id not in available_models:
        raise HTTPException(status_code=404, detail="Model not found")

    return ModelInfo(id=model_id, created=int(time.time()), owned_by="mlx")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI Chat Completions API
    创建聊天完成请求（非流式）
    """
    request_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())

    try:
        # 格式化消息为prompt
        prompt = format_messages(request.messages)

        # 创建sampler - 修复：top_k 必须传递 int，不能是 None
        temp_val = request.temperature if request.temperature is not None else 0.0
        top_p_val = request.top_p if request.top_p is not None else 1.0
        top_k_val = (
            request.top_k if request.top_k is not None else 0
        )  # 必须是0，不能是None！

        sampler = make_sampler(
            temp=temp_val if temp_val > 0 else 0.0,
            top_p=top_p_val if top_p_val > 0 else 1.0,
            top_k=(
                top_k_val if top_k_val and top_k_val > 0 else 0
            ),  # 必须是0，不能是None！
        )

        # 生成响应
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens,
            sampler=sampler,
            verbose=True,
        )

        # 处理 think 标签
        cleaned = clean_response_with_thinking(response)

        # 估算token数（简化版）
        prompt_tokens = len(prompt.split())
        reasoning_tokens = cleaned["reasoning_tokens"]
        content_tokens = len(cleaned["cleaned_content"].split())
        completion_tokens = reasoning_tokens + content_tokens
        total_tokens = prompt_tokens + completion_tokens

        # 构建Usage
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        # 如果有推理内容，添加详细信息
        if cleaned["has_reasoning"]:
            usage.completion_tokens_details = {
                "reasoning_tokens": reasoning_tokens,
                "accepted_prediction_tokens": content_tokens,
                "rejected_prediction_tokens": 0,
            }

        # 构建Response Message
        message = ChatMessage(role="assistant", content=cleaned["cleaned_content"])

        # 如果有推理内容，添加到扩展字段
        message_dict = message.model_dump()
        if cleaned["has_reasoning"]:
            message_dict["reasoning_content"] = cleaned["reasoning_content"]

        # 构建响应
        return ChatCompletionResponse(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0, message=ChatMessage(**message_dict), finish_reason="stop"
                )
            ],
            usage=usage,
        )

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions/stream")
async def stream_chat_completions(request: ChatCompletionRequest):
    """
    OpenAI Chat Completions API
    创建聊天完成请求（流式）
    """
    request_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())

    try:
        # 格式化消息为prompt
        prompt = format_messages(request.messages)

        # 创建sampler - 修复：top_k 必须传递 int，不能是 None
        temp_val = request.temperature if request.temperature is not None else 0.0
        top_p_val = request.top_p if request.top_p is not None else 1.0
        top_k_val = (
            request.top_k if request.top_k is not None else 0
        )  # 必须是0，不能是None！

        sampler = make_sampler(
            temp=temp_val if temp_val > 0 else 0.0,
            top_p=top_p_val if top_p_val > 0 else 1.0,
            top_k=(
                top_k_val if top_k_val and top_k_val > 0 else 0
            ),  # 必须是0，不能是None！
        )

        # 生成响应
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens,
            sampler=sampler,
            verbose=False,
        )

        # 流式返回
        return StreamingResponse(
            openai_stream_generator(response, request_id, created, request.model),
            media_type="text/event-stream",
        )

    except Exception as e:
        logger.error(f"Error generating stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "model": model_path}


# ==================== 启动脚本 ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
