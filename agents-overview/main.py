from typing import Union, List, Optional

from langchain.agents import initialize_agent, tool, AgentType
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain.tools.render import render_text_description

from callbacks import AgentCallbackHandler


@tool
def get_length_text(text: str) -> int:
    """Returns the length of a text by characters."""
    print(f"[get_length_text] Word {text} has {len(text)} characters.")
    return len(text)


@tool
def is_magic_number(number: int) -> bool:
    """Returns if a given number is magic or not."""
    print(f"[is_magic_number] Analyzing if {number} is magic...")
    return number == 42


def use_initialize_agent(query: str):
    print("Using Agent created with langchain.agents.initialize_agent...")
    print("-------------------------------------------------------------")
    llm = ChatOpenAI(callbacks=[AgentCallbackHandler()])
    agent_executor = initialize_agent(
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=[get_length_text, is_magic_number],
        verbose=True,
    )
    agent_executor.invoke({"input": query})


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for t in tools:
        if tool_name == t.name:
            return t
    raise ValueError(f"Tool with name {tool_name} not found in {tools}")


def use_custom_agent(query: str):
    print("Using Custom Agent...")
    print("-------------------------------------------------------------")

    tools = [get_length_text, is_magic_number]
    template = """
        Answer the following questions as best you can. You have access to the following tools:
        {tools}
        
        Use the following format:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Begin!
        
        Question: {input}
        Thought:{agent_scratchpad}
    """
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    llm = ChatOpenAI(
        temperature=0, stop=["\nObservation"], callbacks=[AgentCallbackHandler()]
    )
    intermediate_steps = []

    # using LCEL = Langchain Expression Language
    agent: Union[AgentAction, AgentFinish] = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )
    agent_step: Optional[Union[AgentAction, AgentFinish]] = None
    while not isinstance(agent_step, AgentFinish):
        agent_step = agent.invoke(
            {"input": query, "agent_scratchpad": intermediate_steps}
        )
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input
            observation = tool_to_use.func(str(tool_input))
            intermediate_steps.append((agent_step, observation))
    result = agent_step.return_values["output"]
    print(f"{result}")


if __name__ == "__main__":
    q = "Is the length of the word DOG a magic number?"
    use_custom_agent(q)
    # use_initialize_agent(q)
