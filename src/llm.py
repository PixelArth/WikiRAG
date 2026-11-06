from abc import ABC, abstractmethod
from typing import Literal
import requests

"""
Wrappers to load a LLM model and generate answers from it.
LLM is an abstract class : a subclass is defined for each API and sets its own parameters (temperature, effort...) in its constructor.
Every subclass is then used the same way : llm.call_LLM(prompt, documents).
"""


class LLMError(Exception):
    """
    Raised when the API of a LLM fails (network, rate limit, invalid request...)
    """


class LLM(ABC):
    """
    Abstract class of the LLM wrappers. One instance represents one conversation with a specified
    - system_prompt
    - model used
    - api_key

    call_LLM is a template method : it builds the message sent to the model, then delegates the call of the API to the subclass (_send).
    A subclass must define _send and describe.
    """

    def __init__(self, key=None,
                system_prompt : str | None = "You're an assistant ...",
                model = None):
        self.api_key = key
        self.system_prompt = system_prompt
        self.model = model
        self.usage = {"input_tokens" : 0, "output_tokens" : 0} # tokens consumed by all the calls

    def call_LLM(self, prompt : str, documents : list[dict] | None = None) -> str:
        """
        Sends the prompt to the model and returns its answer. Raises LLMError when the API fails
        - prompt : user message
        - documents : documents the LLM must use to answer the user request, [{"id", "source" : {"type", "text"}}]. A judge needs none
        """
        content = prompt
        if documents is not None:
            content += "\n" + self.formatDocuments(documents)
        return self._send(content)

    @abstractmethod
    def _send(self, content : str) -> str:
        """
        Sends the user message to the API of the subclass with its own parameters and returns the text of the answer
        Must raise LLMError (and no other exception) if the API fails, and sum the tokens consumed with _addUsage
        """

    @abstractmethod
    def describe(self) -> dict:
        """
        Gives the provider, the model and the parameters of the generation, to trace the settings of an evaluation
        """

    @staticmethod
    def formatDocuments(documents : list[dict]) -> str:
        """
        Writes the documents ({"id", "source" : {"type", "text"}}) as the text added to the user message
        """
        return "".join(f"<id>{doc['id']}</id>\n<document>{doc['source']['text']}</document>\n" for doc in documents)

    def _addUsage(self, inputTokens : int, outputTokens : int) -> None:
        """
        Sums the tokens consumed by a call in self.usage
        """
        self.usage["input_tokens"] += inputTokens
        self.usage["output_tokens"] += outputTokens


class Groq(LLM):
    """
    Class communicating with the Groq API : https://api.groq.com/openai/v1/chat/completions
    One instance represents one conversation with a specified
    - system_prompt
    - model use (only 2 openai models from 2024)
    - api_key
    and the parameters of the generation :
    - temperature : temperature of the LLM response
    - max_token : max number of completion tokens
    - effort : effort of the model, gpt-oss-20b and gpt-oss-120b supports 'low', 'medium', 'high'. Not applied to the other models

    Note that the API references a parameter to generate n different response from the LLM but the parameter range is 1 - 1
    """

    _OPENAI_MODELS = ["openai/gpt-oss-20b","openai/gpt-oss-120b"]
    _QWEN_MODELS = ["qwen/qwen3.8-27b"]


    def __init__(self,
                 key=None,
                 system_prompt : str | None = "You're an assistant ...",
                 model : Literal["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"] = "openai/gpt-oss-20b",
                 temperature : float = 1.0,
                 max_token : int = 512,
                 effort : Literal["low","medium","high"] | None = None):
        super().__init__(key=key, system_prompt=system_prompt, model=model)
        self.temperature = temperature
        self.max_token = max_token
        self.effort = effort

        if effort is not None and model not in Groq._OPENAI_MODELS:
            print(f"WARNING : effort has been specified but Groq API only applies this parameter to openai model.\n \
                  {model} is currently in use, the parameter won't have any effect in this session.")

    def _send(self, content : str) -> str:
        """
        Calls the Groq API with the parameters of the instance
        """
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": f"{self.model}",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content}
            ],
            "temperature": self.temperature,
            "max_completion_tokens" : self.max_token,
        }

        if self.model in Groq._OPENAI_MODELS and self.effort is not None:
            payload["reasoning_effort"] = self.effort

        try:
            r = requests.post(url, headers=headers, json=payload)
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(f"Groq API error : {e}") from e

        data = r.json()
        usage = data.get("usage", {})
        self._addUsage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)) # Here would be the usage incrementation if using paid tokens (would not recommend)
        return data["choices"][0]["message"]["content"]

    def describe(self) -> dict:
        return {"provider" : "Groq", "model" : self.model, "temperature" : self.temperature, "max_token" : self.max_token, "effort" : self.effort}


class Claude(LLM):
    """
    Class communicating with the Anthropic API through the official SDK (pip install anthropic) : https://platform.claude.com/docs
    One instance represents one conversation with a specified
    - system_prompt
    - model used : only the lightest and cheapest one, Claude Haiku 5.5 (0.10$ / 1M input tokens, 0.50$ / 1M output tokens, for prompts up to 100K tokens)
    - api_key : None reads the environment variable ANTHROPIC_API_KEY
    and the parameters of the generation :
    - max_token : max number of generated tokens, the thinking tokens are included. A too low value can leave no room for the answer
    - effort : depth of the thinking and overall token spend, 'low' | 'medium' | 'high' | 'xhigh' | 'max'. None uses the default of the model ('medium')
    - thinking : False turns the thinking off (cheaper), only possible with an effort of 'high' or lower

    The Anthropic API rejects the temperature for this model, so there is none.
    """

    def __init__(self,
                 key=None,
                 system_prompt : str | None = "You're an assistant ...",
                 model : Literal["claude-haiku-5-5"] = "claude-haiku-5-5",
                 max_token : int = 1024,
                 effort : Literal["low","medium","high","xhigh","max"] | None = "low",
                 thinking : bool = True):
        super().__init__(key=key, system_prompt=system_prompt, model=model)
        if not thinking and effort in ("xhigh", "max"):
            raise ValueError(f"The thinking can only be turned off with an effort of 'high' or lower, got '{effort}'.")
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("The Claude class needs the Anthropic SDK : pip install anthropic") from e

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=key)
        self.max_token = max_token
        self.effort = effort
        self.thinking = thinking

    def _send(self, content : str) -> str:
        """
        Calls the Anthropic API with the parameters of the instance
        """
        # Reuqest base
        request = {
            "model" : self.model,
            "max_tokens" : self.max_token,
            "messages" : [{"role": "user", "content": content}],
        }
        
        # Add additionnal argument if existing -> system | effort | thinking
        if self.system_prompt is not None:
            request["system"] = self.system_prompt
        if self.effort is not None:
            request["output_config"] = {"effort" : self.effort}
        if not self.thinking:
            request["thinking"] = {"type" : "disabled"}

        # Send request
        try:
            response = self.client.messages.create(**request)
        except self._anthropic.APIError as e:
            raise LLMError(f"Anthropic API error : {e}") from e

        # Add usage and manage stop reasons
        self._addUsage(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason == "max_tokens":
            print(f"WARNING : the answer has been cut at max_token={self.max_token} (the thinking tokens count), increase max_token or lower the effort.")
        elif response.stop_reason == "refusal":
            print("WARNING : the model refused to answer.")

        return "".join(block.text for block in response.content if block.type == "text")

    def describe(self) -> dict:
        return {"provider" : "Claude", "model" : self.model, "max_token" : self.max_token, "effort" : self.effort, "thinking" : self.thinking}
