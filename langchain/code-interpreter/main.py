from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_experimental.agents.agent_toolkits import (
    create_python_agent,
    create_csv_agent,
)
from langchain.chat_models import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool

#import langchain
#langchain.debug = True


def get_python_agent():
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    # python_agent_executor.run(
    #     "Generate and save in current working directory (in a folder named output) 2 qrcodes that point to www.udemy.com/course/langchain."
    #     "You already have the qrcode library installed."
    # )

    return python_agent_executor


def get_csv_agent():
    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="input/episode_info.csv",
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # csv_agent.run("How many columns are in the file input/episode_info.csv?")
    # csv_agent.run("Which writer wrote the most episodes? How many did he/she write?")
    # csv_agent.run("Which season has the most episodes? How many episodes does it have? Use the file input/episode_info.csv")
    return csv_agent


def get_router_agent():
    return initialize_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        tools=[
            Tool(
                name="PythonAgent",
                description="""useful when you need to transform natural language and write from it python and execute the python code,
                                   returning the results of the code execution,
                                   DO NOT SEND PYTHON CODE TO THIS TOOL""",
                func=get_python_agent().run,
            ),
            Tool(
                name="CSVAgent",
                description="""useful when you need to answer question over input/episode_info.csv file,
                                   takes an input the entire question and returns the answer after running pandas calculations""",
                func=get_csv_agent().run,
            ),
        ],
    )


def main():
    agent = get_router_agent()
    agent.run(
        "Generate 2 qrcodes that point to https://www.udemy.com/course/langchain and save the qrcodes in the folder output."
        "You already have the qrcode library installed."
    )
    # agent.run("Print all the seasons and the number of episodes they have in ascending order.")


if __name__ == "__main__":
    main()
