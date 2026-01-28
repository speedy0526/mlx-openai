"""
Anthropic SDK 客户端测试示例
展示如何使用 Anthropic SDK 调用 MLX 服务
"""
import os

# 安装依赖: pip install anthropic
from anthropic import Anthropic

# 配置客户端
client = Anthropic(
    base_url="http://127.0.0.1:8001",  # MLX Anthropic服务地址
    api_key="dummy-key",  # 可以是任意值，因为MLX服务不进行认证
)


def test_message_creation():
    """测试非流式消息创建"""
    print("=" * 60)
    print("测试 1: 非流式消息创建")
    print("=" * 60)
    
    message = client.messages.create(
        model="qwen3-4b",
        max_tokens=256,
        temperature=0.7,
        messages=[
            {"role": "user", "content": "你好，请介绍一下你自己。"}
        ]
    )
    
    print(f"\nResponse ID: {message.id}")
    print(f"Model: {message.model}")
    print(f"Stop reason: {message.stop_reason}")
    print(f"\nContent:")
    for block in message.content:
        if block.type == "text":
            print(block.text)
    print(f"\nUsage:")
    print(f"  Input tokens: {message.usage.input_tokens}")
    print(f"  Output tokens: {message.usage.output_tokens}")


def test_message_streaming():
    """测试流式消息创建"""
    print("\n" + "=" * 60)
    print("测试 2: 流式消息创建")
    print("=" * 60)
    
    with client.messages.stream(
        model="qwen3-4b",
        max_tokens=200,
        temperature=0.8,
        messages=[
            {"role": "user", "content": "写一首关于春天的诗"}
        ]
    ) as stream:
        print("\nResponse (streaming):")
        
        for text in stream.text_stream:
            print(text, end="", flush=True)
        
        print()
        
        # 获取完整消息
        message = stream.get_final_message()
        print(f"\n\nComplete message ID: {message.id}")


def test_system_prompt():
    """测试系统提示"""
    print("\n" + "=" * 60)
    print("测试 3: 系统提示")
    print("=" * 60)
    
    message = client.messages.create(
        model="qwen3-4b",
        max_tokens=150,
        system="你是一个专业的Python编程导师，请用简洁明了的方式回答问题。",
        messages=[
            {"role": "user", "content": "什么是Python装饰器？"}
        ]
    )
    
    print(f"\nContent:")
    for block in message.content:
        if block.type == "text":
            print(block.text)


def test_multi_turn_conversation():
    """测试多轮对话"""
    print("\n" + "=" * 60)
    print("测试 4: 多轮对话")
    print("=" * 60)
    
    conversation = [
        {"role": "user", "content": "我想学习Python，请给我一些建议"},
    ]
    
    print("\nUser: 我想学习Python，请给我一些建议")
    
    message = client.messages.create(
        model="qwen3-4b",
        max_tokens=150,
        messages=conversation
    )
    
    # 获取助手的回复
    assistant_text = ""
    for block in message.content:
        if block.type == "text":
            assistant_text = block.text
            break
    
    print(f"\nAssistant: {assistant_text}")
    
    # 添加助手的回复到对话历史
    conversation.append({"role": "assistant", "content": assistant_text})
    conversation.append({"role": "user", "content": "谢谢你的建议，我现在应该从哪里开始？"})
    
    print("\nUser: 谢谢你的建议，我现在应该从哪里开始？")
    
    message = client.messages.create(
        model="qwen3-4b",
        max_tokens=150,
        messages=conversation
    )
    
    print(f"\nAssistant: ", end="")
    for block in message.content:
        if block.type == "text":
            print(block.text)


def test_stop_sequences():
    """测试停止序列"""
    print("\n" + "=" * 60)
    print("测试 5: 停止序列")
    print("=" * 60)
    
    message = client.messages.create(
        model="qwen3-4b",
        max_tokens=200,
        stop_sequences=["\n\n", "END"],
        messages=[
            {"role": "user", "content": "列出3个学习Python的好处"}
        ]
    )
    
    print(f"\nContent (will stop at \\n\\n or END):")
    for block in message.content:
        if block.type == "text":
            print(block.text)
    print(f"\nStop reason: {message.stop_reason}")


def test_list_models():
    """测试列出模型"""
    print("\n" + "=" * 60)
    print("测试 6: 列出可用模型")
    print("=" * 60)
    
    # 使用原始请求
    import httpx
    
    with httpx.Client(base_url="http://127.0.0.1:8001/v1") as http_client:
        response = http_client.get("/models")
        models_data = response.json()
        
        print("\nAvailable models:")
        for model in models_data["data"]:
            print(f"  - {model['id']} ({model['display_name']})")


def test_temperature_and_top_p():
    """测试温度和top_p参数"""
    print("\n" + "=" * 60)
    print("测试 7: 温度和top_p参数")
    print("=" * 60)
    
    # 低温度（更确定性）
    print("\n--- 低温度 (temperature=0.2) ---")
    message1 = client.messages.create(
        model="qwen3-4b",
        max_tokens=50,
        temperature=0.2,
        messages=[
            {"role": "user", "content": "1+1等于多少？"}
        ]
    )
    
    for block in message1.content:
        if block.type == "text":
            print(block.text)
    
    # 高温度（更多样性）
    print("\n--- 高温度 (temperature=0.9) ---")
    message2 = client.messages.create(
        model="qwen3-4b",
        max_tokens=50,
        temperature=0.9,
        messages=[
            {"role": "user", "content": "写一个有趣的短句"}
        ]
    )
    
    for block in message2.content:
        if block.type == "text":
            print(block.text)


if __name__ == "__main__":
    try:
        # 运行所有测试
        test_list_models()
        test_message_creation()
        test_message_streaming()
        test_system_prompt()
        test_multi_turn_conversation()
        test_stop_sequences()
        test_temperature_and_top_p()
        
        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("=" * 60)
        print("\n提示: 确保安装了 anthropic 包:")
        print("  pip install anthropic")
        print("\n确保 MLX Anthropic 服务正在运行:")
        print("  python anthropic_server.py")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请确保:")
        print("1. MLX Anthropic 服务正在运行 (python anthropic_server.py)")
        print("2. 已安装 anthropic 包 (pip install anthropic)")
