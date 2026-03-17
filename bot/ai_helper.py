from __future__ import annotations

import base64
import datetime
import logging
from abc import ABC, abstractmethod

import openai
import anthropic


class AIHelper(ABC):
    """Base class for AI provider helpers."""

    def __init__(self, config: dict):
        self.config = config
        self.conversations: dict[int, list] = {}
        self.last_updated: dict[int, datetime.datetime] = {}
        self.system_prompt = (
            f"{config['assistant_prompt']}\n\n"
            f"[Provider: {config['provider']}, Model: {config['model']}]"
        )

    async def get_chat_response(self, chat_id: int, query: str):
        """Stream a chat response. Yields (content, 'not_finished') or (content, token_str)."""
        if chat_id not in self.conversations or self._max_age_reached(chat_id):
            self.reset_chat_history(chat_id)

        self.last_updated[chat_id] = datetime.datetime.now()

        if len(self.conversations[chat_id]) > self.config['max_history_size']:
            logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
            try:
                summary = await self._summarise(self.conversations[chat_id][:-1])
                self.reset_chat_history(chat_id)
                self._add_to_history(chat_id, role="user", content="Summarise our conversation")
                self._add_to_history(chat_id, role="assistant", content=summary)
            except Exception as e:
                logging.warning(f'Error while summarising: {e}. Trimming history instead.')
                self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

        self._add_to_history(chat_id, role="user", content=query)

        async for content, tokens in self._stream_response(chat_id):
            yield content, tokens

    @abstractmethod
    async def _stream_response(self, chat_id: int):
        """Provider-specific streaming implementation."""
        ...

    @abstractmethod
    async def _summarise(self, conversation: list) -> str:
        """Summarise conversation history."""
        ...

    @abstractmethod
    async def interpret_image(self, chat_id: int, fileobj, prompt: str | None = None):
        """Interpret an image. Yields (content, 'not_finished') or (content, token_str)."""
        ...

    async def transcribe(self, filename: str) -> str | None:
        """Transcribe audio to text. Returns None if unsupported by this provider."""
        return None

    def supports_transcription(self) -> bool:
        return False

    def reset_chat_history(self, chat_id: int, content: str = ''):
        self.conversations[chat_id] = []

    def _max_age_reached(self, chat_id: int) -> bool:
        if chat_id not in self.last_updated:
            return False
        max_age = self.config['max_conversation_age_minutes']
        return self.last_updated[chat_id] < datetime.datetime.now() - datetime.timedelta(minutes=max_age)

    def _add_to_history(self, chat_id: int, role: str, content: str):
        self.conversations[chat_id].append({"role": role, "content": content})


class OpenAIHelper(AIHelper):
    """OpenAI API helper."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.client = openai.AsyncOpenAI(api_key=config['api_key'])

    def supports_transcription(self) -> bool:
        return True

    async def transcribe(self, filename: str) -> str | None:
        with open(filename, "rb") as audio:
            result = await self.client.audio.transcriptions.create(model="whisper-1", file=audio)
            return result.text

    async def _stream_response(self, chat_id: int):
        messages = [{"role": "system", "content": self.system_prompt}] + self.conversations[chat_id]
        stream = await self.client.chat.completions.create(
            model=self.config['model'],
            messages=messages,
            max_completion_tokens=self.config['max_tokens'],
            stream=True,
        )
        answer = ''
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                answer += delta.content
                yield answer, 'not_finished'

        self._add_to_history(chat_id, role="assistant", content=answer.strip())
        yield answer.strip(), '0'

    async def _summarise(self, conversation: list) -> str:
        conv_text = "\n".join([f"{m['role']}: {m['content']}" for m in conversation])
        response = await self.client.chat.completions.create(
            model=self.config['model'],
            messages=[{"role": "user", "content": f"Summarize this conversation in 700 characters or less:\n\n{conv_text}"}],
        )
        return response.choices[0].message.content.strip()

    async def interpret_image(self, chat_id: int, fileobj, prompt: str | None = None):
        prompt = prompt or self.config['vision_prompt']
        image_data = base64.b64encode(fileobj.getvalue()).decode('utf-8')

        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
        ]}]

        stream = await self.client.chat.completions.create(
            model=self.config['model'],
            messages=messages,
            max_completion_tokens=self.config['max_tokens'],
            stream=True,
        )
        answer = ''
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                answer += delta.content
                yield answer, 'not_finished'

        self._add_to_history(chat_id, role="assistant", content=answer.strip())
        yield answer.strip(), '0'


class AnthropicHelper(AIHelper):
    """Anthropic API helper."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.client = anthropic.AsyncAnthropic(api_key=config['api_key'])

    async def _stream_response(self, chat_id: int):
        async with self.client.messages.stream(
            model=self.config['model'],
            messages=self.conversations[chat_id],
            max_tokens=self.config['max_tokens'],
            system=self.system_prompt,
        ) as stream:
            answer = ''
            usage = 0
            async for chunk in stream:
                if chunk.type == 'content_block_delta':
                    answer += chunk.delta.text
                    yield answer, 'not_finished'
                if chunk.type == 'message_delta':
                    usage += chunk.usage.output_tokens

            self._add_to_history(chat_id, role="assistant", content=answer.strip())
            yield answer.strip(), str(usage)

    async def _summarise(self, conversation: list) -> str:
        conv_text = "\n".join([f"{m['role']}: {m['content']}" for m in conversation])
        response = await self.client.messages.create(
            model=self.config['model'],
            max_tokens=self.config['max_tokens'],
            messages=[{"role": "user", "content": f"Summarize this conversation in 700 characters or less:\n\n{conv_text}"}],
        )
        return response.content[0].text.strip()

    async def interpret_image(self, chat_id: int, fileobj, prompt: str | None = None):
        prompt = prompt or self.config['vision_prompt']
        image_data = base64.b64encode(fileobj.getvalue()).decode('utf-8')

        content = [
            {"type": "text", "text": prompt},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
        ]

        stream = await self.client.messages.create(
            model=self.config['model'],
            max_tokens=self.config['max_tokens'],
            messages=[{"role": "user", "content": content}],
            stream=True,
        )

        answer = ''
        usage = 0
        async for chunk in stream:
            if chunk.type == 'content_block_delta':
                answer += chunk.delta.text
                yield answer, 'not_finished'
            if chunk.type == 'message_delta':
                usage += chunk.usage.output_tokens

        self._add_to_history(chat_id, role="assistant", content=answer.strip())
        yield answer.strip(), str(usage)


def create_ai_helper(config: dict) -> AIHelper:
    """Factory to create the right AI helper based on provider config."""
    provider = config['provider']
    if provider == 'openai':
        return OpenAIHelper(config)
    elif provider == 'anthropic':
        return AnthropicHelper(config)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")
