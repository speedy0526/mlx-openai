"""
OpenAI SDK 客户端测试示例
展示如何使用 OpenAI SDK 调用 MLX 服务
"""
import os
from openai import OpenAI

# 配置客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",  # MLX 服务地址
    api_key="dummy-key",  # 可以是任意值，因为 MLX 服务不进行认证
)


def test_chat_completion():
    """测试非流式聊天完成"""
    print("=" * 60)
    print("测试 1: 非流式聊天完成")
    print("=" * 60)
    
    response = client.chat.completions.create(
        model="qwen3-4b",
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": "你好，请介绍一下你自己。"}
        ],
        temperature=0.7,
        max_tokens=256
    )
    
    print(f"\nResponse:")
    print(response.choices[0].message.content)
    print(f"\nUsage:")
    print(f"  Prompt tokens: {response.usage.prompt_tokens}")
    print(f"  Completion tokens: {response.usage.completion_tokens}")
    print(f"  Total tokens: {response.usage.total_tokens}")


def test_stream_chat_completion():
    """测试流式聊天完成"""
    print("\n" + "=" * 60)
    print("测试 2: 流式聊天完成")
    print("=" * 60)
    
    stream = client.chat.completions.create(
        model="qwen3-4b",
        messages=[
            {"role": "user", "content": "写一首关于春天的诗"}
        ],
        temperature=0.8,
        max_tokens=200,
        stream=True
    )
    
    print("\nResponse (streaming):")
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def test_list_models():
    """测试列出模型"""
    print("\n" + "=" * 60)
    print("测试 3: 列出可用模型")
    print("=" * 60)
    
    models = client.models.list()
    print("\nAvailable models:")
    for model in models.data:
        print(f"  - {model.id} (owned by: {model.owned_by})")


def test_multi_turn_conversation():
    """测试多轮对话"""
    print("\n" + "=" * 60)
    print("测试 4: 多轮对话")
    print("=" * 60)
    
    conversation = [
        {"role": "user", "content": "我想学习Python，请给我一些建议"},
    ]
    
    print("\nUser: 我想学习Python，请给我一些建议")
    
    response = client.chat.completions.create(
        model="qwen3-4b",
        messages=conversation,
        max_tokens=150
    )
    
    assistant_msg = response.choices[0].message.content
    print(f"\nAssistant: {assistant_msg}")
    
    # 添加助手的回复到对话历史
    conversation.append({"role": "assistant", "content": assistant_msg})
    conversation.append({"role": "user", "content": "谢谢你的建议，我现在应该从哪里开始？"})
    
    print("\nUser: 谢谢你的建议，我现在应该从哪里开始？")
    
    response = client.chat.completions.create(
        model="qwen3-4b",
        messages=conversation,
        max_tokens=150
    )
    
    print(f"\nAssistant: {response.choices[0].message.content}")


if __name__ == "__main__":
    try:
        test_list_models()
        test_chat_completion()
        test_stream_chat_completion()
        test_multi_turn_conversation()
        
        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("请确保 MLX 服务正在运行 (python server.py)")
