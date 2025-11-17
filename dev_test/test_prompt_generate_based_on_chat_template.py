import sys
from typing import List, Optional
from dataclasses import dataclass

from sglang.srt.parser.conversation import chat_templates, generate_chat_conv


@dataclass
class ImageURL:
    url: str
    detail: str = "auto"


@dataclass
class ContentPart:
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None
    modalities: str = "image"


@dataclass
class Message:
    role: str
    content: any  # str or List[ContentPart]


@dataclass
class ChatCompletionRequest:
    model: str
    messages: List[Message]
    continue_final_message: bool = False


def test_qwen2_vl_prompt():
    """测试 Qwen2-VL 的 prompt 生成"""
    
    print("=" * 80)
    print("测试 Qwen2.5-VL Chat Template Prompt 生成")
    print("=" * 80)
    
    # 1. 确保 qwen2-vl 模板已注册
    if "qwen2-vl" not in chat_templates:
        raise RuntimeError("\n⚠️  qwen2-vl 模板未注册")
    else:
        print("\n✅ qwen2-vl 模板已存在")
    
    # 2. 构造测试请求
    # from https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/qwen3_vl.md
    request = ChatCompletionRequest(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=[
            Message(
                role="user",
                content=[
                    ContentPart(type="text", text="描述这张图片"),
                    ContentPart(
                        type="image_url",
                        image_url=ImageURL(url="https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"),
                        modalities="image"
                    ),
                ],
            )
        ],
    )
    
    print("\n" + "=" * 80)
    print("输入请求:")
    print("=" * 80)
    print(f"Model: {request.model}")
    print(f"Messages:")
    for msg in request.messages:
        print(f"  - Role: {msg.role}")
        if isinstance(msg.content, list):
            print(f"    Content:")
            for part in msg.content:
                if part.type == "text":
                    print(f"      - Text: {part.text}")
                elif part.type == "image_url":
                    print(f"      - Image URL: {part.image_url.url}")
    
    # 3. 调用 generate_chat_conv 生成 Conversation 对象
    print("\n" + "=" * 80)
    print("步骤 1: 调用 generate_chat_conv()")
    print("=" * 80)
    
    template_name = "qwen2-vl"
    conv = generate_chat_conv(request, template_name)
    
    print(f"✅ Conversation 对象创建成功")
    print(f"  - Template Name: {conv.name}")
    print(f"  - System Message: {conv.system_message}")
    print(f"  - Roles: {conv.roles}")
    print(f"  - Separator: {repr(conv.sep)}")
    print(f"  - Sep Style: {conv.sep_style}")
    print(f"  - Image Token: {conv.image_token}")
    print(f"  - Messages Count: {len(conv.messages)}")
    
    print(f"\n  Messages:")
    for i, (role, content) in enumerate(conv.messages):
        print(f"    [{i}] Role: {role}")
        print(f"        Content: {repr(content)}")
    
    if conv.image_data:
        print(f"\n  Image Data:")
        for i, img in enumerate(conv.image_data):
            print(f"    [{i}] {img}")
    
    # 4. 调用 get_prompt() 生成最终 prompt
    print("\n" + "=" * 80)
    print("步骤 2: 调用 conv.get_prompt()")
    print("=" * 80)
    
    prompt = conv.get_prompt()
    
    print("✅ Prompt 生成成功\n")
    print("生成的 Prompt:")
    print("-" * 80)
    print(prompt)
    print("-" * 80)
    
    # 5. 验证 prompt 格式
    print("\n" + "=" * 80)
    print("步骤 3: 验证 Prompt 格式")
    print("=" * 80)
    
    expected_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
描述这张图片<|vision_start|><|image_pad|><|vision_end|><|im_end|>
<|im_start|>assistant
"""
    
    print("\n期望的 Prompt:")
    print("-" * 80)
    print(expected_prompt)
    print("-" * 80)
    
    # 逐行比较
    print("\n逐行比较:")
    print("-" * 80)
    
    prompt_lines = prompt.split('\n')
    expected_lines = expected_prompt.split('\n')
    
    max_lines = max(len(prompt_lines), len(expected_lines))
    all_match = True
    
    for i in range(max_lines):
        actual = prompt_lines[i] if i < len(prompt_lines) else "<缺失>"
        expected = expected_lines[i] if i < len(expected_lines) else "<缺失>"
        
        match = actual == expected
        all_match = all_match and match
        
        status = "✅" if match else "❌"
        print(f"行 {i+1:2d} {status}")
        print(f"  实际: {repr(actual)}")
        print(f"  期望: {repr(expected)}")
        if not match:
            print(f"  差异: 不匹配!")
        print()
    
    # 6. 最终结果
    print("=" * 80)
    if all_match:
        print("🎉 测试通过！生成的 prompt 与期望格式完全一致！")
    else:
        print("❌ 测试失败！生成的 prompt 与期望格式不一致！")
    print("=" * 80)
    
    # 7. 额外检查关键元素
    print("\n" + "=" * 80)
    print("额外检查:")
    print("=" * 80)
    
    checks = [
        ("包含系统消息开始标记", "<|im_start|>system" in prompt),
        ("包含系统消息内容", "You are a helpful assistant." in prompt),
        ("包含用户消息开始标记", "<|im_start|>user" in prompt),
        ("包含文本内容", "描述这张图片" in prompt),
        ("包含图像 token", "<|vision_start|><|image_pad|><|vision_end|>" in prompt),
        ("包含消息结束标记", "<|im_end|>" in prompt),
        ("包含助手消息开始标记", "<|im_start|>assistant" in prompt),
        ("以助手标记结尾", prompt.strip().endswith("<|im_start|>assistant")),
    ]
    
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
    
    return all_match


def test_multimodal_variations():
    """测试多种多模态场景"""
    
    print("\n\n" + "=" * 80)
    print("测试多种场景")
    print("=" * 80)
    
    # 场景 1: 纯文本
    print("\n场景 1: 纯文本消息")
    print("-" * 80)
    request1 = ChatCompletionRequest(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=[
            Message(role="user", content="你好，请介绍一下你自己")
        ],
    )
    conv1 = generate_chat_conv(request1, "qwen2-vl")
    prompt1 = conv1.get_prompt()
    print(prompt1)
    
    # 场景 2: 多张图片
    print("\n场景 2: 多张图片")
    print("-" * 80)
    request2 = ChatCompletionRequest(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=[
            Message(
                role="user",
                content=[
                    ContentPart(type="text", text="比较这两张图片的区别"),
                    ContentPart(
                        type="image_url",
                        image_url=ImageURL(url="https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"),
                        modalities="image"
                    ),
                    ContentPart(
                        type="image_url",
                        image_url=ImageURL(url="https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"),
                        modalities="image"
                    ),
                ],
            )
        ],
    )
    conv2 = generate_chat_conv(request2, "qwen2-vl")
    prompt2 = conv2.get_prompt()
    print(prompt2)
    
    # 场景 3: 图片在文本前面
    print("\n场景 3: 图片在文本前面")
    print("-" * 80)
    request3 = ChatCompletionRequest(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=[
            Message(
                role="user",
                content=[
                    ContentPart(
                        type="image_url",
                        image_url=ImageURL(url="https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"),
                        modalities="image"
                    ),
                    ContentPart(type="text", text="这是什么动物？"),
                ],
            )
        ],
    )
    conv3 = generate_chat_conv(request3, "qwen2-vl")
    prompt3 = conv3.get_prompt()
    print(prompt3)
    
    # 场景 4: 多轮对话
    print("\n场景 4: 多轮对话")
    print("-" * 80)
    request4 = ChatCompletionRequest(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=[
            Message(
                role="user",
                content=[
                    ContentPart(type="text", text="描述这张图片"),
                    ContentPart(
                        type="image_url",
                        image_url=ImageURL(url="https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"),
                        modalities="image"
                    ),
                ],
            ),
            Message(role="assistant", content="这是一只可爱的小狗。"),
            Message(role="user", content="它是什么品种？"),
        ],
    )
    conv4 = generate_chat_conv(request4, "qwen2-vl")
    prompt4 = conv4.get_prompt()
    print(prompt4)


if __name__ == "__main__":
    try:
        success = test_qwen2_vl_prompt()
        
        test_multimodal_variations()
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
