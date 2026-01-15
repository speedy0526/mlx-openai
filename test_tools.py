"""
测试模型使用 tools (函数调用) 的能力
"""
from openai import OpenAI

# 配置客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key",
)


# 定义可用的工具
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、纽约"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位",
                        "default": "celsius"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行基本的数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如：2 + 3, 10 * 5, 100 / 4"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "在数据库中搜索信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数量限制",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# 工具函数实现
def get_weather(city: str, unit: str = "celsius") -> dict:
    """模拟获取天气信息"""
    weather_data = {
        "北京": {"temp": 25, "condition": "晴", "humidity": 45},
        "上海": {"temp": 28, "condition": "多云", "humidity": 65},
        "纽约": {"temp": 15, "condition": "雨", "humidity": 80},
        "伦敦": {"temp": 12, "condition": "阴", "humidity": 70},
        "东京": {"temp": 20, "condition": "晴", "humidity": 55}
    }
    
    if city in weather_data:
        data = weather_data[city]
        if unit == "fahrenheit":
            data["temp"] = data["temp"] * 9/5 + 32
        return {
            "city": city,
            "temperature": data["temp"],
            "unit": unit,
            "condition": data["condition"],
            "humidity": data["humidity"]
        }
    else:
        return {
            "city": city,
            "error": "未找到该城市的天气数据"
        }


def calculate(expression: str) -> dict:
    """执行数学计算"""
    try:
        # 注意：eval 有安全风险，生产环境应使用更安全的方法
        result = eval(expression)
        return {
            "expression": expression,
            "result": result,
            "status": "success"
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "status": "error"
        }


def search_database(query: str, limit: int = 10) -> dict:
    """模拟数据库搜索"""
    # 模拟数据库
    mock_db = [
        {"id": 1, "title": "Python编程入门", "category": "编程"},
        {"id": 2, "title": "机器学习基础", "category": "AI"},
        {"id": 3, "title": "深度学习实战", "category": "AI"},
        {"id": 4, "title": "Web开发指南", "category": "编程"},
        {"id": 5, "title": "数据结构", "category": "计算机科学"},
        {"id": 6, "title": "算法导论", "category": "计算机科学"},
        {"id": 7, "title": "人工智能历史", "category": "AI"},
        {"id": 8, "title": "JavaScript高级", "category": "编程"},
    ]
    
    results = [
        item for item in mock_db
        if query.lower() in item["title"].lower() or query.lower() in item["category"].lower()
    ]
    
    return {
        "query": query,
        "total": len(results),
        "limit": limit,
        "results": results[:limit]
    }


# 工具调用映射
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_database": search_database
}


def extract_tool_calls(response_text: str) -> list:
    """
    从响应文本中提取工具调用
    这是一个简单的解析器，用于从文本中识别类似函数调用的模式
    """
    import re
    
    # 查找类似函数调用的模式: function_name(arg1=value1, arg2=value2) 或 function_name(value)
    pattern = r'(\w+)\((.*?)\)'
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    tool_calls = []
    for match in matches:
        func_name, args_str = match
        
        # 只处理我们定义的工具
        if func_name not in TOOL_FUNCTIONS:
            continue
            
        # 解析参数
        try:
            args = {}
            if args_str.strip():
                # 检查是否是关键字参数格式: key=value
                if '=' in args_str:
                    # 使用正则表达式提取关键字参数
                    # 匹配: key="value", key='value', key=123, key=true
                    arg_pattern = r'(\w+)\s*=\s*(["\'](.*?)["\']|(\d+(?:\.\d+)?)|true|false|null)'
                    arg_matches = re.findall(arg_pattern, args_str)
                    
                    for key, quoted_value1, unquoted_value2, number_value in arg_matches:
                        if quoted_value1:
                            # 有引号的字符串
                            args[key] = unquoted_value2
                        elif number_value:
                            # 数字
                            try:
                                args[key] = int(number_value)
                            except ValueError:
                                args[key] = float(number_value)
                        elif quoted_value1 == 'true':
                            args[key] = True
                        elif quoted_value1 == 'false':
                            args[key] = False
                        elif quoted_value1 == 'null':
                            args[key] = None
                else:
                    # 位置参数 - 根据函数名推断参数
                    # 清理参数字符串（去除引号、空格等）
                    arg_value = args_str.strip()
                    # 去除外层引号
                    if (arg_value.startswith('"') and arg_value.endswith('"')) or \
                       (arg_value.startswith("'") and arg_value.endswith("'")):
                        arg_value = arg_value[1:-1]
                    
                    if func_name == 'get_weather':
                        args['city'] = arg_value
                    elif func_name == 'calculate':
                        # 可能是表达式，保持原样
                        args['expression'] = args_str.strip()
                    elif func_name == 'search_database':
                        args['query'] = arg_value
            
            if args:  # 只有成功解析到参数才添加
                tool_calls.append({
                    "function": {
                        "name": func_name,
                        "arguments": args
                    }
                })
        except Exception as e:
            print(f"解析工具调用失败 '{func_name}({args_str})': {e}")
            continue
    
    return tool_calls


def test_tool_calling():
    """测试工具调用功能"""
    print("=" * 70)
    print("测试 1: 简单工具调用 - 获取天气")
    print("=" * 70)
    
    messages = [
        {
            "role": "system",
            "content": "你是一个智能助手，可以使用提供的工具来帮助用户。当需要使用工具时，请使用以下格式调用工具：function_name(arg1=value1, arg2=value2)"
        },
        {
            "role": "user",
            "content": "北京的天气怎么样？"
        }
    ]
    
    response = client.chat.completions.create(
        model="qwen3-4b",
        messages=messages,
        temperature=0.7,
        max_tokens=256
    )
    
    response_text = response.choices[0].message.content
    print(f"\n模型响应:\n{response_text}")
    
    # 尝试提取工具调用
    tool_calls = extract_tool_calls(response_text)
    print(f"\n检测到的工具调用: {len(tool_calls)}")
    
    if tool_calls:
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            args = tool_call["function"]["arguments"]
            print(f"\n调用函数: {func_name}")
            print(f"参数: {args}")
            
            if func_name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[func_name](**args)
                print(f"执行结果: {result}")
    
    messages.append({
        "role": "assistant",
        "content": response_text
    })


def test_multi_tool_calling():
    """测试多个工具调用"""
    print("\n" + "=" * 70)
    print("测试 2: 多个工具调用 - 数学计算")
    print("=" * 70)
    
    messages = [
        {
            "role": "system",
            "content": "你是一个智能助手，可以使用提供的工具来帮助用户。当需要使用工具时，请使用以下格式调用工具：function_name(arg1=value1, arg2=value2)"
        },
        {
            "role": "user",
            "content": "帮我计算 25 * 4 和 100 / 5 的结果"
        }
    ]
    
    response = client.chat.completions.create(
        model="qwen3-4b",
        messages=messages,
        temperature=0.3,
        max_tokens=300
    )
    
    response_text = response.choices[0].message.content
    print(f"\n模型响应:\n{response_text}")
    
    # 尝试提取工具调用
    tool_calls = extract_tool_calls(response_text)
    print(f"\n检测到的工具调用: {len(tool_calls)}")
    
    if tool_calls:
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            args = tool_call["function"]["arguments"]
            print(f"\n调用函数: {func_name}")
            print(f"参数: {args}")
            
            if func_name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[func_name](**args)
                print(f"执行结果: {result}")


def test_conversation_with_tools():
    """测试带工具的多轮对话"""
    print("\n" + "=" * 70)
    print("测试 3: 带工具的多轮对话")
    print("=" * 70)
    
    messages = [
        {
            "role": "system",
            "content": "你是一个智能助手，可以使用提供的工具来帮助用户。当需要使用工具时，请使用以下格式调用工具：function_name(arg1=value1, arg2=value2)"
        },
        {
            "role": "user",
            "content": "我想搜索关于机器学习的资料"
        }
    ]
    
    # 第一轮
    response = client.chat.completions.create(
        model="qwen3-4b",
        messages=messages,
        temperature=0.5,
        max_tokens=200
    )
    
    response_text = response.choices[0].message.content
    print(f"\n用户: 我想搜索关于机器学习的资料")
    print(f"\n助手: {response_text}")
    
    # 提取并执行工具调用
    tool_calls = extract_tool_calls(response_text)
    
    if tool_calls:
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            args = tool_call["function"]["arguments"]
            
            if func_name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[func_name](**args)
                print(f"\n工具执行结果: {result}")
                
                # 将工具结果添加到对话
                messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                messages.append({
                    "role": "tool",
                    "name": func_name,
                    "content": str(result)
                })
    
    # 第二轮：基于工具结果的回应
    response = client.chat.completions.create(
        model="qwen3-4b",
        messages=messages,
        temperature=0.5,
        max_tokens=200
    )
    
    response_text = response.choices[0].message.content
    print(f"\n助手: {response_text}")


def test_direct_tool_description():
    """测试直接告诉模型工具描述"""
    print("\n" + "=" * 70)
    print("测试 4: 直接在prompt中描述工具")
    print("=" * 70)
    
    tools_description = """
你可以使用以下工具：

1. get_weather(city, unit='celsius') - 获取城市天气
   - city: 城市名称（必需）
   - unit: 温度单位，celsius 或 fahrenheit（可选，默认celsius）

2. calculate(expression) - 执行数学计算
   - expression: 数学表达式（必需），例如：2 + 3, 10 * 5

3. search_database(query, limit=10) - 搜索数据库
   - query: 搜索查询（必需）
   - limit: 返回结果数量（可选，默认10）

调用工具时请使用格式：函数名(参数1=值1, 参数2=值2)
"""
    
    messages = [
        {
            "role": "system",
            "content": f"你是一个智能助手。{tools_description}"
        },
        {
            "role": "user",
            "content": "计算 100 + 200，然后告诉我上海的天气"
        }
    ]
    
    response = client.chat.completions.create(
        model="qwen3-4b",
        messages=messages,
        temperature=0.5,
        max_tokens=300
    )
    
    response_text = response.choices[0].message.content
    print(f"\n模型响应:\n{response_text}")
    
    # 提取并执行工具调用
    tool_calls = extract_tool_calls(response_text)
    print(f"\n检测到的工具调用: {len(tool_calls)}")
    
    if tool_calls:
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            args = tool_call["function"]["arguments"]
            print(f"\n调用函数: {func_name}")
            print(f"参数: {args}")
            
            if func_name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[func_name](**args)
                print(f"执行结果: {result}")


if __name__ == "__main__":
    try:
        test_tool_calling()
        test_multi_tool_calling()
        test_conversation_with_tools()
        test_direct_tool_description()
        
        print("\n" + "=" * 70)
        print("所有工具测试完成!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请确保 MLX 服务正在运行 (python server.py)")
