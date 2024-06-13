from enum import Enum
import requests as r


class LlmModels(Enum):
    LLAMA3_8B_INSTRUCT = 'llama3:instruct'
    LLAMA3_70B_INSTRUCT = 'llama3:70b-instruct'
    LLAMA3_70B = 'llama3:70b'

class OllamaClient:
    def __init__(self, model_name: LlmModels=LlmModels.LLAMA3_70B_INSTRUCT, temperature: float = 0., host: str = '127.0.0.1:11434'):
        self.model = model_name.value
        self.temperature = temperature
        self.host = host
        self.system_prompt = """Need to generate a prompt in the format 'Object in the background'.  
Always generate 10 prompt variants, the response should only contain numbered prompts without an introduction.  
Example: 'Battery in the living room with blue walls.'  
The request provides an object and its category. You need to add a background and a preposition. Choose relevant backgrounds for the object so that the image described could be found on a marketplace selling furniture.  
More prompt examples:  
'Garden swing in the yard of a country house,'  
'Lamp on a white ceiling in a living room with a sofa.'"""
        

    def generate(self, prompt: str) -> str:
        body = dict(
            model=self.model,
            prompt=prompt,
            system=self.system_prompt,
            stream=False,
            temperature=self.temperature,
        )
        resp = r.post(f'http://{self.host}/api/generate', json=body)
        assert resp.status_code == 200, resp.text

        data = resp.json()['response']
        return data


if __name__ == "__main__":
    model = OllamaClient(model_name=LlmModels.LLAMA3_8B_INSTRUCT)
    custom_prompt = model.generate(f'garden swing, garden furniture')