from typing import Literal
from langchain_core.documents import Document

import requests



"""
Wrapper to load a LLM model and generate answers from it.
Subclasses can be defined for any model API if needed.
"""


class LLM:
    def __init__(self, key=None, 
                system_prompt : str | None = "You're an assistant ...",
                model = None,):
        self.api_key = key
        self.system_prompt = system_prompt
        self.model = model

    def call_LLM(self, prompt : str, **kwargs):
        pass

class Groq(LLM):
    """
    Class communicating with the Groq API : https://api.groq.com/openai/v1/chat/completions
    One instance represents one conversation with a specified 
    - system_prompt 
    - model use (only 2 openai models from 2024)
    - api_key
    """
    
    
    def __init__(self, 
                 key=None, 
                 system_prompt : str | None = "You're an assistant ...",
                 model : Literal["openai/gpt-oss-120b", "openai/gpt-oss-20b"] | None = "openai/gpt-oss-20b"):
        super().__init__(key=key, system_prompt=system_prompt, model=model)

    def call_LLM(self, 
                prompt: str, 
                temperature : float | None = 1.0,
                documents : list[dict] | None = None, 
                max_token : int | None = 512,
                effort : Literal["low","medium","high"] | None = "medium"
                ) -> str:

        """
        Calls groq API with the desired parameters : 
        - prompt : user message
        - effort : effort of the model, gpt-oss-20b and gpt-oss-120b supports 'low', 'medium', 'high'
        - temperarature : temperature of the LLM response
        - documents : documents the LLM must use to answer the user request
        - max_tokens : max number of completion tokens
        - n_response : number of different responses given by the model
        
        Note that the API references a parameter to generate n different response from the LLM but the parameter range is 1 - 1
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
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_completion_tokens" : max_token,
            "reasoning_effort" : effort
        }
        
                
        if documents is not None:
            #self.check_document_format(documents=documents)
            all_documents = "\n"
            
            for doc in documents:
                id = doc["id"]
                text = doc["source"]["text"]
                all_documents += f"<id>{id}<id>\n<document>{text}<document>\n"
            
            payload["messages"][1]["content"] += all_documents
            
        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def check_document_format(self,documents : list[dict]) -> None:
        
        main_keys = {"id","source"}
        text_object_keys = {"text","type"}

        for doc in documents:
            for k,v in doc.items():
                if not k in main_keys:
                    raise ValueError(f"key of the documents must be in {main_keys}. Found '{k}'.")
                if k == "source":
                    if isinstance(v, dict):
                        for text_k,text_v in v.items():
                            if text_k not in text_object_keys:
                                raise ValueError(f"key of the documents must be in {main_keys}. Found '{text_k}'.")
                            if text_k == "type":
                                if text_v != "text":
                                    raise ValueError("Text type must be 'text'")
                            else:
                                if not isinstance(text_v, str):
                                    raise ValueError("Text content must be str")
                    else:
                        raise ValueError(f"Bad object found for key {k  }, expected dict, got : {v}")
                else:
                    if not (v is None or isinstance(v,str)):
                        raise ValueError(f"Expected str or None, got {v}")