import re
import subprocess
import webbrowser
from typing import List

from llama_index.llms import OpenAI
from llama_index.agent import ReActAgent
from llama_index.tools import FunctionTool
from llama_index.callbacks import LlamaDebugHandler, CallbackManager

llm = OpenAI()


def write_haiku(topic: str) -> str:
    """
    Writes a haiku about a given topic
    """
    return llm.complete(f"Write me a haiku about {topic}").text


def count_characters(text: str) -> int:
    """
    Count the number of characters in a text
    """
    return len(text)


def break_text_into_words(text: str) -> List[str]:
    """ "
    Breaks a text into words
    """
    return re.findall(r"\b\w+\b", text)


def open_application(application_name: str) -> str:
    """ "
    Opens an application in my computer
    """
    try:
        subprocess.Popen(["start", application_name], shell=True)
        return "successfully opened the application"
    except Exception as e:
        print(f"Error: {str(e)}")


def open_url(url: str) -> str:
    """
    Opens an url in the default user's browser (like Chrome / Safari / Firefox)
    """
    try:
        webbrowser.open(url)
        return "successfully opened the url in the default browser"
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])

    agent = ReActAgent.from_tools(
        llm=llm,
        tools=[
            FunctionTool.from_defaults(fn=write_haiku, name="Write haiku"),
            FunctionTool.from_defaults(fn=count_characters, name="Count characters"),
            FunctionTool.from_defaults(
                fn=break_text_into_words, name="Break text into words"
            ),
            FunctionTool.from_defaults(fn=open_application, name="Open app"),
            FunctionTool.from_defaults(fn=open_url, name="Open url"),
        ],
        verbose=True,
        callback_manager=callback_manager,
    )
    # response = agent.query("Write me a haiku about tennis and then count the characters in it.")
    # response = agent.query("Open notepad in my computer.")
    response = agent.query("Open https://openai.com/ in my browser.")
    # response = agent.query(
    #     "Write me a haiku about artificial intelligence, "
    #     "Break it into words, "
    #     "Count the amount of characters of each word, "
    #     "Take the biggest word found and "
    #     "Open an url searching for this word on Google"
    # )
    print(response)
