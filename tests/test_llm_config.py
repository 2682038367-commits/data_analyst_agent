"""模型供应商配置与客户端构造测试（不调用外部 API）。"""
from __future__ import annotations

import unittest
from unittest.mock import patch, sentinel

from src.config import resolve_fallback_settings, resolve_llm_settings


class LlmConfigTests(unittest.TestCase):
    def test_explicit_deepseek_keeps_existing_overrides(self) -> None:
        provider, key, base_url, model, _ = resolve_llm_settings({
            "LLM_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "deepseek-test",
            "LLM_BASE_URL": "https://custom.example/v1",
            "LLM_MODEL": "custom-model",
        })
        self.assertEqual(provider, "deepseek")
        self.assertEqual(key, "deepseek-test")
        self.assertEqual(base_url, "https://custom.example/v1")
        self.assertEqual(model, "custom-model")

    def test_glm_uses_its_own_key_model_and_endpoint(self) -> None:
        provider, key, base_url, model, hint = resolve_llm_settings({
            "LLM_PROVIDER": "glm",
            "DEEPSEEK_API_KEY": "wrong-provider-key",
            "LLM_BASE_URL": "https://api.deepseek.com",
            "LLM_MODEL": "deepseek-chat",
            "GLM_API_KEY": "glm-test",
        })
        self.assertEqual(provider, "glm")
        self.assertEqual(key, "glm-test")
        self.assertEqual(base_url, "https://open.bigmodel.cn/api/paas/v4/")
        self.assertEqual(model, "glm-5.3")
        self.assertIn("GLM_API_KEY", hint)

    def test_glm_accepts_official_zai_key_and_overrides(self) -> None:
        provider, key, base_url, model, _ = resolve_llm_settings({
            "LLM_PROVIDER": "GLM",
            "ZAI_API_KEY": "zai-test",
            "GLM_BASE_URL": "https://example.test/v4/",
            "GLM_MODEL": "glm-custom",
        })
        self.assertEqual(provider, "glm")
        self.assertEqual(key, "zai-test")
        self.assertEqual(base_url, "https://example.test/v4/")
        self.assertEqual(model, "glm-custom")

    def test_unknown_provider_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "LLM_PROVIDER"):
            resolve_llm_settings({"LLM_PROVIDER": "unknown"})

    def test_client_receives_selected_glm_settings(self) -> None:
        from src import graph

        with (
            patch.object(graph, "LLM_API_KEY", "glm-test"),
            patch.object(graph, "LLM_FALLBACK_API_KEY", ""),
            patch.object(graph, "LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"),
            patch.object(graph, "LLM_MODEL", "glm-5.3"),
            patch.object(graph, "_build_http_clients",
                         return_value=(sentinel.http_client, sentinel.async_client)),
            patch.object(graph, "ChatOpenAI") as client_class,
        ):
            graph._get_llm()
        kwargs = client_class.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "glm-test")
        self.assertEqual(kwargs["base_url"], "https://open.bigmodel.cn/api/paas/v4/")
        self.assertEqual(kwargs["model"], "glm-5.3")

    def test_missing_key_fails_before_client_creation(self) -> None:
        from src import graph

        with (
            patch.object(graph, "LLM_API_KEY", ""),
            patch.object(graph, "LLM_FALLBACK_API_KEY", ""),
            patch.object(graph, "LLM_KEY_HINT", "GLM_API_KEY"),
            patch.object(graph, "_build_http_clients") as build_clients,
        ):
            with self.assertRaisesRegex(RuntimeError, "GLM_API_KEY"):
                graph._get_llm()
        build_clients.assert_not_called()


    def test_glm_is_default_primary(self) -> None:
        provider, key, _, model, _ = resolve_llm_settings({
            "GLM_API_KEY": "glm-test",
            "DEEPSEEK_API_KEY": "deepseek-test",
        })
        self.assertEqual((provider, key, model), ("glm", "glm-test", "glm-5.3"))

    def test_deepseek_fallback_has_independent_settings(self) -> None:
        key, url, model = resolve_fallback_settings({
            "DEEPSEEK_API_KEY": "deepseek-test",
            "LLM_BASE_URL": "https://wrong.example/v1",
            "LLM_MODEL": "wrong-model",
        })
        self.assertEqual(key, "deepseek-test")
        self.assertEqual(url, "https://api.deepseek.com")
        self.assertEqual(model, "deepseek-chat")
        self.assertEqual(
            resolve_fallback_settings({"LLM_PROVIDER": "deepseek",
                                       "DEEPSEEK_API_KEY": "deepseek-test"}),
            ("", "", ""),
        )
        self.assertEqual(
            resolve_fallback_settings({"OPENAI_API_KEY": "not-deepseek"})[0], ""
        )

    def test_glm_error_uses_deepseek_for_same_request(self) -> None:
        from langchain_core.runnables import RunnableLambda
        from src import graph

        calls = []

        def fail_glm(value):
            calls.append(("glm", value))
            raise RuntimeError("GLM unavailable")

        def use_deepseek(value):
            calls.append(("deepseek", value))
            return "fallback answer"

        with (
            patch.object(graph, "LLM_API_KEY", "glm-test"),
            patch.object(graph, "LLM_FALLBACK_API_KEY", "deepseek-test"),
            patch.object(graph, "LLM_FALLBACK_BASE_URL", "https://api.deepseek.com"),
            patch.object(graph, "LLM_FALLBACK_MODEL", "deepseek-chat"),
            patch.object(graph, "_build_http_clients",
                         return_value=(sentinel.http_client, sentinel.async_client)),
            patch.object(graph, "ChatOpenAI", side_effect=[
                RunnableLambda(fail_glm), RunnableLambda(use_deepseek),
            ]) as client_class,
        ):
            answer = graph._get_llm().invoke("同一个问题")
        self.assertEqual(answer, "fallback answer")
        self.assertEqual(calls, [
            ("glm", "同一个问题"), ("deepseek", "同一个问题"),
        ])
        self.assertEqual(client_class.call_args_list[1].kwargs["model"], "deepseek-chat")
        self.assertEqual(client_class.call_args_list[0].kwargs["api_key"], "glm-test")
        self.assertEqual(client_class.call_args_list[1].kwargs["api_key"], "deepseek-test")
        self.assertEqual(
            client_class.call_args_list[1].kwargs["base_url"],
            "https://api.deepseek.com",
        )

    def test_glm_success_does_not_invoke_deepseek(self) -> None:
        from langchain_core.runnables import RunnableLambda
        from src import graph

        def should_not_run(_):
            raise AssertionError("fallback should not run")

        with (
            patch.object(graph, "LLM_API_KEY", "glm-test"),
            patch.object(graph, "LLM_FALLBACK_API_KEY", "deepseek-test"),
            patch.object(graph, "_build_http_clients",
                         return_value=(sentinel.http_client, sentinel.async_client)),
            patch.object(graph, "ChatOpenAI", side_effect=[
                RunnableLambda(lambda _: "glm answer"),
                RunnableLambda(should_not_run),
            ]),
        ):
            self.assertEqual(graph._get_llm().invoke("问题"), "glm answer")

    def test_missing_glm_key_uses_deepseek_directly(self) -> None:
        from src import graph

        with (
            patch.object(graph, "LLM_API_KEY", ""),
            patch.object(graph, "LLM_FALLBACK_API_KEY", "deepseek-test"),
            patch.object(graph, "LLM_FALLBACK_BASE_URL", "https://api.deepseek.com"),
            patch.object(graph, "LLM_FALLBACK_MODEL", "deepseek-chat"),
            patch.object(graph, "_build_http_clients",
                         return_value=(sentinel.http_client, sentinel.async_client)),
            patch.object(graph, "ChatOpenAI") as client_class,
        ):
            graph._get_llm()
        client_class.assert_called_once()
        self.assertEqual(client_class.call_args.kwargs["api_key"], "deepseek-test")
        self.assertEqual(client_class.call_args.kwargs["model"], "deepseek-chat")

if __name__ == "__main__":
    unittest.main()
