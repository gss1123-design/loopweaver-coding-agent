"""One-request container worker. It does not load credentials or extensions."""
import asyncio
import json
import sys

from ai.types import TextContent, ToolResultMessage
from .builtin_tools import create_builtin_tools
from .serde import message_to_dict


async def main():
    request = json.load(sys.stdin)
    try:
        tools = create_builtin_tools("/workspace", enabled_names=[request["name"]], **request["policy"])
        tool = next(t for t in tools if t.name == request["name"])
        result = await tool.execute(request["call_id"], request["args"], None, None)
        message = ToolResultMessage(tool_call_id=request["call_id"], tool_name=tool.name,
                                    content=result.content, details=result.details)
    except Exception as exc:
        message = ToolResultMessage(is_error=True, content=[TextContent(text=str(exc))])
    print(json.dumps(message_to_dict(message), ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
