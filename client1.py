import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient #this is the library to allow client to connect to multiple MCP servers
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
import json

load_dotenv()

SERVERS = { 
    "math": {
        "transport": "stdio",  # Local server
        "command": "/Library/Frameworks/Python.framework/Versions/3.11/bin/uv",#commands and arg are just instrictions to start the server
        "args": [
            "run",
            "fastmcp",
            "run",
            "/Users/arpan/Desktop/mcp-math-server/main.py"
       ]
    },
    "expense": {
        "transport": "streamable_http",  # if this fails, try "sse"  # Remote server
        "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
    },
    "manim-server": {
        "transport": "stdio",
        "command": "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
        "args": [
        "/Users/arpan/desktop/manim-mcp-server/src/manim_server.py"
      ],
        "env": {
        "MANIM_EXECUTABLE": "/Library/Frameworks/Python.framework/Versions/3.11/bin/manim"
      }
    }
}

async def main():
    
    client = MultiServerMCPClient(SERVERS) # create a client that connects to multiple MCP servers
    tools = await client.get_tools() #to fetch all the tool in local mcp tool


    named_tools = {}
    for tool in tools:
        named_tools[tool.name] = tool # create a dictionary of tool name to tool object contains all info of tool including config and how to call it

    print("Available tools:", named_tools.keys())

    llm = ChatOpenAI(model="gpt-5")  # Initialize the LLM to use tool
    llm_with_tools = llm.bind_tools(tools) # bind the list of tools to the llm 

    prompt = "Draw a triangle rotating in place using the manim tool."
    response = await llm_with_tools.ainvoke(prompt) # this will give response which tool to call and args will be needed

    if not getattr(response, "tool_calls", None): 
        print("\nLLM Reply:", response.content) # If the LLM response has no tool calls, print the direct llm reply and exit
        return

    tool_messages = []  # Initialize a list to store tool execution results as messages
    for tc in response.tool_calls:  # Iterate over each tool call in the LLM response as sometime llm might suggest multiple tool calls 
        selected_tool = tc["name"]  # Get the name of the tool to call
        selected_tool_args = tc.get("args") or {}  # Get the arguments for the tool, defaulting to empty dict if none
        selected_tool_id = tc["id"]  # Get the unique ID of the tool call

        result = await named_tools[selected_tool].ainvoke(selected_tool_args)  # Asynchronously invoke the selected tool with its args
        tool_messages.append(ToolMessage(tool_call_id=selected_tool_id, content=json.dumps(result)))  # Append the result as a ToolMessage with JSON content

        

    final_response = await llm_with_tools.ainvoke([prompt, response, *tool_messages])  # Invoke the LLM again with the original prompt, initial response, and tool results to generate a final answer
    print(f"Final response: {final_response.content}")  # Print the content of the final LLM response




if __name__ == '__main__':
    asyncio.run(main())
