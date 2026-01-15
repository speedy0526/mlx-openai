"""
测试模型返回 <think> 标签的处理方式
"""
from openai import OpenAI

# 配置客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key",
)


def test_think_tag_basic():
    """测试基本 think 标签处理"""
    print("=" * 70)
    print("测试 1: 基本响应（包含 think 标签）")
    print("=" * 70)
    
    response = client.chat.completions.create(
        model="qwen3-4b",
        messages=[
            {
                "role": "user",
                "content": "请一步步思考，计算 25 * 4 + 10 的结果"
            }
        ],
        temperature=0.7,
        max_tokens=300
    )
    
    content = response.choices[0].message.content
    print(f"\n原始响应:\n{content}")
    
    # 处理 <think> 标签的方法
    print("\n" + "-" * 70)
    print("处理方法 1: 移除 <think> 标签内容")
    print("-" * 70)
    
    import re
    
    # 方法1: 完全移除 think 标签及其内容
    cleaned_response = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    print(f"\n清理后的响应:\n{cleaned_response}")
    
    print("\n" + "-" * 70)
    print("处理方法 2: 提取 think 标签内容作为推理过程")
    print("-" * 70)
    
    # 方法2: 分别提取思考过程和最终答案
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    if think_match:
        thought_process = think_match.group(1).strip()
        final_answer = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        print(f"\n思考过程:\n{thought_process}")
        print(f"\n最终答案:\n{final_answer}")
    
    print("\n" + "-" * 70)
    print("处理方法 3: 保留标签但用于前端显示区分")
    print("-" * 70)
    
    # 方法3: 将 think 标签转换为 HTML 用于前端显示
    formatted_response = content.replace(
        '<think>',
        '\n🤔 思考过程:\n```'
    ).replace(
        '</think>',
        '```\n\n💡 回答:\n'
    )
    print(f"\n格式化响应:\n{formatted_response}")


def test_think_tag_with_reasoning():
    """测试需要推理的问题"""
    print("\n" + "=" * 70)
    print("测试 2: 复杂推理问题")
    print("=" * 70)
    
    response = client.chat.completions.create(
        model="qwen3-4b",
        messages=[
            {
                "role": "system",
                "content": "如果需要思考，请使用 <think> 标签包裹你的思考过程，然后在标签外给出最终答案。"
            },
            {
                "role": "user",
                "content": "一个农场里有鸡和兔子，共20个头，56条腿。鸡和兔子各有多少只？请详细说明推理过程。"
            }
        ],
        temperature=0.5,
        max_tokens=400
    )
    
    content = response.choices[0].message.content
    print(f"\n完整响应:\n{content}")
    
    # 提取结构化信息
    import re
    
    structured_data = {
        "has_thinking": False,
        "thought_process": "",
        "final_answer": ""
    }
    
    if '<think>' in content and '</think>' in content:
        structured_data["has_thinking"] = True
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if think_match:
            structured_data["thought_process"] = think_match.group(1).strip()
            structured_data["final_answer"] = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    else:
        structured_data["final_answer"] = content.strip()
    
    print("\n" + "-" * 70)
    print("结构化数据:")
    print("-" * 70)
    print(f"包含思考过程: {structured_data['has_thinking']}")
    print(f"\n思考过程:\n{structured_data['thought_process']}")
    print(f"\n最终答案:\n{structured_data['final_answer']}")


def test_openai_style_response():
    """模拟 OpenAI API 的响应格式（reasoning_content）"""
    print("\n" + "=" * 70)
    print("测试 3: 模拟 OpenAI reasoning API 格式")
    print("=" * 70)
    
    response = client.chat.completions.create(
        model="qwen3-4b",
        messages=[
            {
                "role": "user",
                "content": "解释什么是递归，并给出一个例子。"
            }
        ],
        temperature=0.7,
        max_tokens=400
    )
    
    content = response.choices[0].message.content
    
    # 模拟 OpenAI 的处理方式：将 think 标签内容提取到 reasoning_content 字段
    import re
    
    # 假设的 OpenAI 响应格式
    openai_style_response = {
        "id": response.id,
        "model": response.model,
        "created": response.created,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",  # 这将是清理后的内容
                    "reasoning_content": "" if '<think>' not in content else None  # 思考过程
                },
                "finish_reason": response.choices[0].finish_reason
            }
        ],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }
    
    # 处理 think 标签
    if '<think>' in content and '</think>' in content:
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if think_match:
            openai_style_response["choices"][0]["message"]["reasoning_content"] = think_match.group(1).strip()
            openai_style_response["choices"][0]["message"]["content"] = re.sub(
                r'<think>.*?</think>', '', content, flags=re.DOTALL
            ).strip()
    else:
        openai_style_response["choices"][0]["message"]["content"] = content.strip()
    
    print(f"\n原始响应:\n{content}")
    print("\n" + "-" * 70)
    print("OpenAI 风格响应结构:")
    print("-" * 70)
    
    import json
    print(json.dumps(openai_style_response, ensure_ascii=False, indent=2))


def test_streaming_with_think_tags():
    """测试流式输出中的 think 标签处理"""
    print("\n" + "=" * 70)
    print("测试 4: 流式输出处理 think 标签")
    print("=" * 70)
    
    print("\n正在接收流式响应...")
    
    stream = client.chat.completions.create(
        model="qwen3-4b",
        messages=[
            {
                "role": "user",
                "content": "简单解释量子力学的基本概念"
            }
        ],
        temperature=0.7,
        max_tokens=300,
        stream=True
    )
    
    import re
    
    # 状态跟踪
    in_think_block = False
    thinking_buffer = []
    content_buffer = []
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            
            # 检测 think 标签的开始和结束
            if '<think>' in content:
                in_think_block = True
                content = content.replace('<think>', '')
            
            if '</think>' in content:
                in_think_block = False
                content = content.replace('</think>', '')
            
            # 根据状态分发内容
            if in_think_block:
                thinking_buffer.append(content)
                print(f"🤔 {content}", end='', flush=True)
            else:
                content_buffer.append(content)
                print(f"💡 {content}", end='', flush=True)
    
    full_content = ''.join(content_buffer)
    full_thinking = ''.join(thinking_buffer)
    
    print("\n\n" + "-" * 70)
    print("完整内容:")
    print("-" * 70)
    print(f"思考部分 ({len(full_thinking)} 字符):\n{full_thinking}")
    print(f"\n回答部分 ({len(full_content)} 字符):\n{full_content}")


def test_filter_think_in_server():
    """在服务端过滤 think 标签（推荐做法）"""
    print("\n" + "=" * 70)
    print("测试 5: 服务端过滤 think 标签（推荐）")
    print("=" * 70)
    
    # 注意：这需要在 server.py 中实现
    # 可以添加后处理函数来清理响应
    
    print("\n推荐的实现方式：")
    print("1. 在 server.py 中添加响应后处理函数")
    print("2. 自动检测并处理 <think> 标签")
    print("3. 返回清理后的内容给客户端")
    
    print("\n示例代码（需要在 server.py 中添加）：")
    print("""
def clean_response(text: str) -> dict:
    \"\"\"清理响应文本，提取 think 内容和最终答案\"\"\"
    import re
    
    result = {
        "content": text,
        "reasoning": None
    }
    
    # 检查是否有 think 标签
    if '<think>' in text and '</think>' in text:
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            result["reasoning"] = think_match.group(1).strip()
            result["content"] = re.sub(
                r'<think>.*?</think>', '', text, flags=re.DOTALL
            ).strip()
    
    return result
    
# 在 chat_completions 函数中使用：
cleaned = clean_response(response)
# 返回时可以将 reasoning 放入扩展字段
    """)


if __name__ == "__main__":
    try:
        test_think_tag_basic()
        test_think_tag_with_reasoning()
        test_openai_style_response()
        test_streaming_with_think_tags()
        test_filter_think_in_server()
        
        print("\n" + "=" * 70)
        print("所有 think 标签测试完成!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
