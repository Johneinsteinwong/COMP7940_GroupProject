#import configparser
import os
import requests
from langchain_core.language_models.chat_models import BaseChatModel, ChatGeneration
from langchain.schema import ChatResult, AIMessage, HumanMessage, BaseMessage, SystemMessage, LLMResult, Generation

from pydantic import BaseModel, Field
from typing import Any, Dict


# Convert LangChain messages to a JSON-serializable format
def message_to_dict(msg: BaseMessage):
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    elif isinstance(msg, AIMessage):
        return {"role": "assistant", "content": msg.content}
    elif isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    else:
        raise TypeError(f"Unsupported message type: {type(msg)}")
        
class HKBU_ChatGPT(BaseChatModel):
    base_url: str = Field(...)
    model: str = Field(...)
    api_version: str = Field(...)
    api_key: str = Field(...)

    def __init__(self, base_url, model, api_version, api_key, **kwargs):
        super().__init__(base_url=base_url, model=model, api_version=api_version, api_key=api_key, **kwargs)

    def _generate(self, message, stop=None, **kwargs) -> ChatResult:

        conversation = [message_to_dict(msg) for msg in message]
        print(type(message),message)

        url = self.base_url + \
        "/deployments/" + self.model + \
        "/chat/completions/?api-version=" + self.api_version

        header = {
            'Content-Type' : 'application/json',
            'api-key' : self.api_key
        }

        payload = {'messages':  conversation}
        response = requests.post(url, headers=header, json=payload)
        if response.status_code == 200:
            data = response.json()
            assistant_reply = data['choices'][0]['message']['content']

            message = AIMessage(content=assistant_reply)
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
        else:
            return 'Error:', response
        
    @property
    def _llm_type(self):
        """Return model type."""
        return "custom_HKBU_chat"
        
