import os
import unittest
from unittest.mock import patch

from src.llm import FallbackChatModel, content_to_text, get_llm


class FakeChatModel:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def invoke(self, messages, *args, **kwargs):
        self.calls.append((messages, args, kwargs))
        if self.error:
            raise self.error
        return self.response


class LlmFactoryTests(unittest.TestCase):
    def test_content_to_text_accepts_plain_string(self):
        self.assertEqual(content_to_text("hello"), "hello")

    def test_content_to_text_flattens_gemini_content_parts(self):
        content = [
            {"type": "text", "text": "```python\nprint('ok')\n```"},
            {"type": "text", "text": "ignored prose"},
        ]

        self.assertEqual(
            content_to_text(content),
            "```python\nprint('ok')\n```\nignored prose",
        )

    def test_fallback_chat_model_uses_primary_when_it_succeeds(self):
        primary = FakeChatModel(response="gemini response")
        fallback = FakeChatModel(response="qwen response")
        model = FallbackChatModel(primary=primary, fallback=fallback)

        result = model.invoke(["hello"])

        self.assertEqual(result, "gemini response")
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(len(fallback.calls), 0)

    def test_fallback_chat_model_uses_qwen_when_gemini_fails(self):
        primary = FakeChatModel(error=RuntimeError("gemini unavailable"))
        fallback = FakeChatModel(response="qwen response")
        model = FallbackChatModel(primary=primary, fallback=fallback)

        result = model.invoke(["hello"])

        self.assertEqual(result, "qwen response")
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(len(fallback.calls), 1)

    def test_get_llm_uses_environment_config_for_both_providers(self):
        captured = {}

        def google_factory(**kwargs):
            captured["google"] = kwargs
            return FakeChatModel(response="gemini")

        def qwen_factory(**kwargs):
            captured["qwen"] = kwargs
            return FakeChatModel(response="qwen")

        env = {
            "GEMINI_API_KEY": "gemini-test-key",
            "QWEN_API_KEY": "qwen-test-key",
            "GEMINI_MODEL": "gemini-test-model",
            "QWEN_MODEL": "qwen-test-model",
            "QWEN_BASE_URL": "https://qwen.example/v1",
            "LLM_REQUEST_TIMEOUT_SECONDS": "45",
            "LLM_MAX_RETRIES": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            model = get_llm(
                temperature=0.3,
                google_factory=google_factory,
                qwen_factory=qwen_factory,
            )

        self.assertIsInstance(model, FallbackChatModel)
        self.assertEqual(captured["google"]["model"], "gemini-test-model")
        self.assertEqual(captured["google"]["api_key"], "gemini-test-key")
        self.assertEqual(captured["google"]["temperature"], 0.3)
        self.assertEqual(captured["google"]["timeout"], 45.0)
        self.assertEqual(captured["google"]["max_retries"], 1)
        self.assertEqual(captured["qwen"]["model"], "qwen-test-model")
        self.assertEqual(captured["qwen"]["api_key"], "qwen-test-key")
        self.assertEqual(captured["qwen"]["base_url"], "https://qwen.example/v1")
        self.assertEqual(captured["qwen"]["temperature"], 0.3)
        self.assertEqual(captured["qwen"]["timeout"], 45.0)
        self.assertEqual(captured["qwen"]["max_retries"], 1)

    def test_get_llm_returns_qwen_directly_when_gemini_construction_fails(self):
        def google_factory(**kwargs):
            raise RuntimeError("gemini package missing")

        fallback = FakeChatModel(response="qwen")

        def qwen_factory(**kwargs):
            return fallback

        env = {"GEMINI_API_KEY": "gemini-test-key", "QWEN_API_KEY": "qwen-test-key"}
        with patch.dict(os.environ, env, clear=True):
            model = get_llm(google_factory=google_factory, qwen_factory=qwen_factory)

        self.assertIs(model, fallback)


if __name__ == "__main__":
    unittest.main()
