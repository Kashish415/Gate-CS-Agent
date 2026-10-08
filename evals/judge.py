import os
from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model="openai/gpt-oss-120b"):
        self._model = model
        api_key = os.getenv("GROQ_API_KEY")
        self._client = Groq(api_key=api_key) if api_key else Groq()

    def get_model_name(self):
        return self._model

    def load_model(self):
        return self._model

    def generate(self, prompt, schema=None):
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"} if schema else None,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)
