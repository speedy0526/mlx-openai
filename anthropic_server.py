"""
Anthropic API-compatible MLX inference server
提供与 Anthropic Claude API 兼容的消息接口
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局模型和tokenizer
model = None
tokenizer = None
model_path = "./models/qwen2-3b"


# ==================== 数据模型 ====================


class TextContentBlock(BaseModel):
    """文本内容块"""

    type: Literal["text"] = Field(default="text", description="内容块类型")
    text: str = Field(..., description="文本内容")


class ImageContentBlock(BaseModel):
    """图片内容块（预留，暂不实现）"""

    type: Literal["image"] = Field(default="image", description="内容块类型")
    source: Dict[str, Any] = Field(..., description="图片源信息")


ContentBlock = Annotated[Union[TextContentBlock, ImageContentBlock], Field(discriminator="type")]


class Message(BaseModel):
    """消息"""

    role: Literal["user", "assistant"] = Field(..., description="消息角色")
    content: Union[str, List[ContentBlock]] = Field(..., description="消息内容")

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v):
        """验证 content 字段，确保字符串不会被错误解析"""
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            # 验证每个内容块
            validated_blocks = []
            for block in v:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "text":
                        validated_blocks.append(TextContentBlock(**block))
                    elif block_type == "image":
                        validated_blocks.append(ImageContentBlock(**block))
                    else:
                        raise ValueError(f"Unsupported content block type: {block_type}")
                else:
                    validated_blocks.append(block)
            return validated_blocks
        raise ValueError(f"Content must be a string or a list of content blocks, got {type(v)}")


class MessageRequest(BaseModel):
    """Anthropic Messages API 请求"""

    model: str = Field(default="qwen2-3b", description="模型名称")
    messages: List[Message] = Field(..., description="消息列表")
    max_tokens: int = Field(default=16384, ge=1, description="最大生成token数")
    temperature: float = Field(default=0.1, ge=0.0, description="采样温度")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="top_p采样")
    top_k: Optional[int] = Field(default=None, ge=0, description="top_k采样")
    stop_sequences: Optional[List[str]] = Field(default=None, description="停止序列")
    stream: bool = Field(default=False, description="是否流式输出")
    system: Optional[Union[str, List[ContentBlock]]] = Field(
        default=None, description="系统提示"
    )

    @field_validator("system", mode="before")
    @classmethod
    def validate_system(cls, v):
        """验证 system 字段，支持字符串和 content blocks"""
        if v is None:
            return v
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            # 验证每个内容块
            validated_blocks = []
            for block in v:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "text":
                        validated_blocks.append(TextContentBlock(**block))
                    elif block_type == "image":
                        validated_blocks.append(ImageContentBlock(**block))
                    else:
                        raise ValueError(f"Unsupported system content block type: {block_type}")
                else:
                    validated_blocks.append(block)
            return validated_blocks
        raise ValueError(f"System must be a string or a list of content blocks, got {type(v)}")


class Usage(BaseModel):
    """Token使用情况"""

    input_tokens: int = Field(default=0, description="输入token数")
    output_tokens: int = Field(default=0, description="输出token数")
    cache_creation_input_tokens: Optional[int] = Field(
        default=0, description="缓存创建输入token"
    )
    cache_read_input_tokens: Optional[int] = Field(
        default=0, description="缓存读取输入token"
    )


class MessageResponse(BaseModel):
    """Anthropic Messages API 响应"""

    id: str = Field(..., description="响应ID")
    type: Literal["message"] = Field(default="message", description="响应类型")
    role: Literal["assistant"] = Field(default="assistant", description="角色")
    content: List[Dict[str, Any]] = Field(..., description="内容块列表")
    model: str = Field(..., description="模型名称")
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence"]] = Field(
        default="end_turn", description="停止原因"
    )
    stop_sequence: Optional[str] = Field(default=None, description="停止序列")
    usage: Usage = Field(..., description="Token使用情况")


class MessageStreamEvent(BaseModel):
    """流式响应事件基类"""

    type: str = Field(..., description="事件类型")


class MessageStartEvent(MessageStreamEvent):
    """消息开始事件"""

    type: Literal["message_start"] = "message_start"
    message: Dict[str, Any] = Field(..., description="消息信息")


class MessageDeltaEvent(MessageStreamEvent):
    """消息增量事件"""

    type: Literal["message_delta"] = "message_delta"
    delta: Dict[str, Optional[str]] = Field(..., description="增量信息")
    usage: Dict[str, int] = Field(..., description="Token使用情况")


class ContentBlockStartEvent(MessageStreamEvent):
    """内容块开始事件"""

    type: Literal["content_block_start"] = "content_block_start"
    index: int = Field(..., description="内容块索引")
    content_block: Dict[str, Any] = Field(..., description="内容块信息")


class ContentBlockDeltaEvent(MessageStreamEvent):
    """内容块增量事件"""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int = Field(..., description="内容块索引")
    delta: Dict[str, str] = Field(..., description="增量文本")


class ContentBlockStopEvent(MessageStreamEvent):
    """内容块停止事件"""

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int = Field(..., description="内容块索引")


class MessageStopEvent(MessageStreamEvent):
    """消息停止事件"""

    type: Literal["message_stop"] = "message_stop"


class ModelInfo(BaseModel):
    """模型信息"""

    id: str = Field(..., description="模型ID")
    display_name: Optional[str] = Field(default=None, description="显示名称")
    version: Optional[str] = Field(default=None, description="版本")
    created_at: Optional[str] = Field(default=None, description="创建时间")


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
    title="MLX Anthropic-Compatible API",
    description="Anthropic-compatible API for MLX inference",
    version="0.1.0",
    lifespan=lifespan,
)


# ==================== 辅助函数 ====================


def clean_response_with_thinking(text: str) -> Dict[str, Any]:
    """
    清理响应文本，提取 think 标签内容和最终答案

    Args:
        text: 原始响应文本

    Returns:
        dict: 包含 cleaned_content（清理后的内容）和
              reasoning_content（推理过程，如果有的话）
    """
    result = {
        "cleaned_content": text,
        "reasoning_content": None,
        "has_reasoning": False,
    }

    # 处理 think 标签
    start_pattern = r"<think>"
    end_pattern = r"</think>"

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

    # 清理多余空行
    result["cleaned_content"] = re.sub(
        r"\n{3,}", "\n\n", result["cleaned_content"]
    ).strip()

    return result


def format_anthropic_messages(
    messages: List[Message], system: Optional[Union[str, List[ContentBlock]]] = None
) -> str:
    """将Anthropic消息列表格式化为prompt字符串"""
    chat_messages = []

    # 添加系统消息（如果有）
    if system:
        if isinstance(system, str):
            chat_messages.append({"role": "system", "content": system})
        else:
            # 处理 system 的 content blocks
            text_parts = []
            for block in system:
                if isinstance(block, TextContentBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ImageContentBlock):
                    # 暂不支持图片，跳过
                    pass
            if text_parts:
                chat_messages.append({"role": "system", "content": " ".join(text_parts)})

    # 转换消息格式
    for msg in messages:
        if isinstance(msg.content, str):
            chat_messages.append({"role": msg.role, "content": msg.content})
        else:
            # 处理复杂内容块
            text_parts = []
            for block in msg.content:
                if isinstance(block, TextContentBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ImageContentBlock):
                    # 暂不支持图片，跳过
                    pass
            chat_messages.append({"role": msg.role, "content": " ".join(text_parts)})

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
                prompt += f"Human: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n\n"
            elif msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
        prompt += "Assistant:"
        return prompt


async def anthropic_stream_generator(
    response: str, request_id: str, model_name: str, max_tokens: int
) -> AsyncGenerator[str, None]:
    """Anthropic流式响应生成器"""
    cleaned = clean_response_with_thinking(response)
    content = cleaned["cleaned_content"]

    # 发送 message_start 事件
    start_event = {
        "type": "message_start",
        "message": {
            "id": request_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model_name,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},  # 简化处理
        },
    }
    yield json.dumps(start_event) + "\n"

    # 发送 content_block_start 事件
    block_start_event = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    yield json.dumps(block_start_event) + "\n"

    # 逐字符发送内容
    for char in content:
        delta_event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": char},
        }
        yield json.dumps(delta_event) + "\n"
        await asyncio.sleep(0.01)

    # 发送 content_block_stop 事件
    block_stop_event = {"type": "content_block_stop", "index": 0}
    yield json.dumps(block_stop_event) + "\n"

    # 发送 message_delta 事件
    delta_event = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": len(content.split())},  # 简化计算
    }
    yield json.dumps(delta_event) + "\n"

    # 发送 message_stop 事件
    stop_event = {"type": "message_stop"}
    yield json.dumps(stop_event) + "\n"


# ==================== API 端点 ====================


@app.post("/v1/messages", response_model=MessageResponse)
async def create_message(request: MessageRequest):
    """
    Anthropic Messages API
    创建消息完成请求（非流式）
    """
    request_id = f"msg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    try:
        # 格式化消息为prompt
        prompt = format_anthropic_messages(request.messages, request.system)

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

        # 处理响应
        cleaned = clean_response_with_thinking(response)

        # 估算token数（简化版）
        input_tokens = len(prompt.split())
        output_tokens = len(cleaned["cleaned_content"].split())

        # 构建内容块
        content_blocks = [{"type": "text", "text": cleaned["cleaned_content"]}]

        # 构建响应
        return MessageResponse(
            id=request_id,
            role="assistant",
            content=content_blocks,
            model=request.model,
            stop_reason="end_turn",
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/messages/stream")
async def create_message_stream(request: MessageRequest):
    """
    Anthropic Messages API
    创建消息完成请求（流式）
    """
    request_id = f"msg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    try:
        # 格式化消息为prompt
        prompt = format_anthropic_messages(request.messages, request.system)

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
            anthropic_stream_generator(
                response, request_id, request.model, request.max_tokens
            ),
            media_type="text/event-stream",
        )

    except Exception as e:
        logger.error(f"Error generating stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """列出可用模型（Anthropic风格）"""
    return {
        "data": [
            {
                "id": "qwen2-3b",
                "display_name": "Qwen3 4B",
                "version": "1.0.0",
                "created_at": "2025-01-01T00:00:00.000Z",
            },
            {
                "id": "yuntu-llm-2b",
                "display_name": "Yuntu LLM 2B",
                "version": "1.0.0",
                "created_at": "2025-01-01T00:00:00.000Z",
            },
        ],
        "object": "list",
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """获取指定模型信息"""
    available_models = ["qwen2-3b", "yuntu-llm-2b"]

    if model_id not in available_models:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "id": model_id,
        "display_name": model_id.replace("-", " ").title(),
        "version": "1.0.0",
        "created_at": "2025-01-01T00:00:00.000Z",
        "object": "model",
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "model": model_path}


# ==================== 启动脚本 ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("anthropic_server:app", host="0.0.0.0", port=8001, reload=True)
