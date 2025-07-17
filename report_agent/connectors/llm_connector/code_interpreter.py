# report_agent/connectors/llm_connector/code_interpreter.py

from openai import OpenAI
from typing import List, Dict, Any
from report_agent.connectors.llm_connector.base import LLMConnector

class CodeInterpreterConnector(LLMConnector):
    """
    A connector that uses OpenAI's Code Interpreter (python tool)
    instead of function‑calling.  Everything else (prompt, run_report, etc.)
    stays on the LLMConnector interface.
    """

    def __init__(self,
                 api_key: str,
                 model_name: str = "gpt-4.1",       # or your favourite code‑enabled model
    ):
        super().__init__(api_key, model_name)
        self.client = OpenAI(api_key=api_key)

        # register exactly one tool: the python sandbox
        self.tools = [{
            "type": "code_interpreter",
            "container": {"type": "auto"}
        }]

        # fn_map unused in this connector but we'll keep it empty
        self.fn_map = {}

        # we’ll stash the full sequence of tool steps here for inspection
        self.last_tool_output: List[Any] = []

    def register_tools(self, functions: List[Any]) -> None:
        # not needed for code_interpreter branch
        pass

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        functions: List[Dict[str, Any]],
        function_call: str
    ) -> Dict[str, Any]:
        """
        Instead of chat.completions, we call client.responses.create()
        with our python tool.
        """
        # we take the last user prompt as `input`
        user_input = messages[-1]["content"]

        resp = self.client.responses.create(
            model=self.model_name,
            tools=self.tools,
            instructions=(
                "You are an AI analyst.  "
                "Use the python tool to load, inspect, and plot the data as needed, "
                "then emit a final text report."
            ),
            input=user_input
        )

        # keep the raw sequence so we can inspect code, reasoning, files, etc.
        self.last_tool_output = resp.output

        # wrap it back into the same shape base.run_report expects:
        return {
            "choices": [
                {"message": {"content": resp.output_text}}
            ]
        }
