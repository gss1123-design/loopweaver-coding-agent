from __future__ import annotations

"""
上下文溢出检测模块。

本模块负责在发送请求前估算当前对话上下文的 token 总量，
并根据模型的 context_window 上限判定是否会溢出。
Agent 可以据此在调用 LLM 前主动触发压缩或截断策略，
避免请求被拒绝或产生不可预期的截断行为。

核心能力：
- 支持多模态消息（文本 + 图片）的 token 估算
- 针对中英文混合文本采用差异化启发式估算
- 提供溢出判定和占用比例两种检测模式
"""

import math

from .types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

# ========== 估算常量 ==========
# 英文平均每 token 约 4 字符（GPT 系列 tokenizer 的经验值）
CHARS_PER_TOKEN = 4

# 单张图片估算为 1000 token（Claude 的图片 token 消耗较大，此处保守估计）
IMAGE_TOKEN_ESTIMATE = 1000

# 每个 tool schema 定义估算为 200 token（包括 name、description、parameters）
TOOL_SCHEMA_TOKEN_ESTIMATE = 200


def _estimate_text_tokens(text: str) -> int:
    """
    用保守的中英文混合启发式估算文本 token 数。

    Token 估算策略：
    - 中文字符（CJK）：每个字符约 0.6 token
      （因为中文在大多数 tokenizer 中会被拆分为多个 byte，但不到 1:1）
    - 英文及其他字符：每 4 个字符约 1 token（CHARS_PER_TOKEN = 4）

    Unicode 范围覆盖：
    - U+3400..U+4DBF: CJK 扩展 A
    - U+4E00..U+9FFF: CJK 统一表意文字（常用汉字）
    - U+F900..U+FAFF: CJK 兼容表意文字

    Args:
        text: 待估算的文本字符串

    Returns:
        估算的 token 数量，最小为 1
    """
    # 统计 CJK 字符数量
    cjk_chars = sum(
        1
        for char in text
        if (
            "㐀" <= char <= "䶿"  # CJK 扩展 A
            or "一" <= char <= "鿿"  # CJK 统一表意文字
            or "豈" <= char <= "﫿"  # CJK 兼容表意文字
        )
    )

    # 其余字符（英文、标点、空格等）
    other_chars = len(text) - cjk_chars

    # 混合计算：中文 0.6x + 英文 0.25x，向上取整
    return max(1, math.ceil(cjk_chars * 0.6 + other_chars / CHARS_PER_TOKEN))


def estimate_message_tokens(msg: Message) -> int:
    """
    估算单条消息的 token 数量。

    根据消息类型（UserMessage、AssistantMessage、ToolResultMessage）
    遍历其内容块（TextContent、ImageContent、ThinkingContent、ToolCall），
    分别累加各部分的 token 估算值。

    Args:
        msg: 待估算的消息对象

    Returns:
        该消息的估算 token 数，最小为 1
    """
    total = 0

    # 用户消息：可能是纯文本字符串，或包含文本 + 图片的 content block 列表
    if isinstance(msg, UserMessage):
        if isinstance(msg.content, str):
            total += _estimate_text_tokens(msg.content)
        else:
            for block in msg.content:
                if isinstance(block, TextContent):
                    total += _estimate_text_tokens(block.text)
                elif isinstance(block, ImageContent):
                    total += IMAGE_TOKEN_ESTIMATE

    # 助手消息：包含文本输出、思考内容（thinking）、工具调用等
    elif isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextContent):
                total += _estimate_text_tokens(block.text)
            elif isinstance(block, ThinkingContent):
                total += _estimate_text_tokens(block.thinking)
            elif isinstance(block, ToolCall):
                # 工具调用需计入：函数名 + 参数 JSON + 固定开销（约 20 token）
                total += _estimate_text_tokens(str(block.arguments))
                total += _estimate_text_tokens(block.name) + 20

    # 工具结果消息：可能包含文本或图片（如截图、生成的图表等）
    elif isinstance(msg, ToolResultMessage):
        for block in msg.content:
            if isinstance(block, TextContent):
                total += _estimate_text_tokens(block.text)
            elif isinstance(block, ImageContent):
                total += IMAGE_TOKEN_ESTIMATE

    return max(1, total)


def estimate_context_tokens(
    messages: list[Message],
    system_prompt: str = "",
    tools: list | None = None,
) -> int:
    """
    估算整个上下文（system prompt + 消息历史 + tool schemas）的 token 总量。

    Args:
        messages: 对话消息列表
        system_prompt: 系统提示词（可选）
        tools: 工具定义列表（可选）

    Returns:
        上下文总 token 估算值
    """
    # 1. 系统提示词
    total = _estimate_text_tokens(system_prompt) if system_prompt else 0

    # 2. 所有历史消息
    for msg in messages:
        total += estimate_message_tokens(msg)

    # 3. 工具 schema 定义（每个工具约 200 token）
    if tools:
        total += len(tools) * TOOL_SCHEMA_TOKEN_ESTIMATE

    return total


def is_context_overflow(
    model: Model,
    context: Context,
    *,
    safety_margin: float = 0.95,
) -> bool:
    """
    检查 context 是否超出模型 context_window。

    为避免模型因上下文过长拒绝请求，这里引入安全系数：
    默认 0.95 表示只使用 95% 的窗口，留 5% 给模型输出和系统开销。

    Args:
        model: 模型对象（需包含 context_window 属性）
        context: 当前对话上下文
        safety_margin: 安全系数，默认 0.95（即保留 5% 余量）

    Returns:
        True 表示已溢出，需要压缩；False 表示尚未溢出
    """
    # 计算安全上限
    limit = int(model.context_window * safety_margin)

    # 估算当前 token 用量
    estimated = estimate_context_tokens(
        context.messages,
        context.system_prompt or "",
        context.tools,
    )

    return estimated > limit


def overflow_ratio(model: Model, context: Context) -> float:
    """
    返回当前 token 占 context_window 的比例。

    用于监控和日志记录，便于观察上下文占用趋势。
    例如：0.8 表示已用 80%，0.95 接近临界点，1.2 表示已超出 20%。

    Args:
        model: 模型对象
        context: 当前对话上下文

    Returns:
        占用比例（0.0 ~ 正无穷），0.0 表示空上下文或窗口未定义
    """
    estimated = estimate_context_tokens(
        context.messages,
        context.system_prompt or "",
        context.tools,
    )

    # 避免除零
    if model.context_window <= 0:
        return 0.0

    return estimated / model.context_window
