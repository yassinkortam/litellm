import importlib.util
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy import free_threading as ft
from litellm.proxy.proxy_cli import ProxyInitializationHelpers


# --------------------------------------------------------------------------- #
# free_threading detection helpers
# --------------------------------------------------------------------------- #
class TestFreeThreadingDetection:
    def test_gil_disabled_build_true(self):
        with patch.object(ft.sysconfig, "get_config_var", return_value=1):
            assert ft.is_gil_disabled_build() is True

    def test_gil_disabled_build_false(self):
        with patch.object(ft.sysconfig, "get_config_var", return_value=0):
            assert ft.is_gil_disabled_build() is False

    def test_is_gil_enabled_none_when_unsupported(self):
        # Standard interpreters (<3.13) have no sys._is_gil_enabled.
        with patch.object(ft.sys, "_is_gil_enabled", None, create=True):
            assert ft.is_gil_enabled() is None

    def test_is_gil_enabled_reports_runtime_state(self):
        with patch.object(ft.sys, "_is_gil_enabled", lambda: False, create=True):
            assert ft.is_gil_enabled() is False
        with patch.object(ft.sys, "_is_gil_enabled", lambda: True, create=True):
            assert ft.is_gil_enabled() is True

    def test_is_free_threaded_requires_build_and_disabled_gil(self):
        # Free-threaded build + GIL off => True
        with (
            patch.object(ft, "is_gil_disabled_build", return_value=True),
            patch.object(ft, "is_gil_enabled", return_value=False),
        ):
            assert ft.is_free_threaded() is True
        # Free-threaded build but GIL re-enabled => False
        with (
            patch.object(ft, "is_gil_disabled_build", return_value=True),
            patch.object(ft, "is_gil_enabled", return_value=True),
        ):
            assert ft.is_free_threaded() is False
        # Standard build => False
        with patch.object(ft, "is_gil_disabled_build", return_value=False):
            assert ft.is_free_threaded() is False

    def test_status_message_variants(self):
        with patch.object(ft, "is_free_threaded", return_value=True):
            assert "GIL disabled" in ft.free_threading_status()
        with (
            patch.object(ft, "is_free_threaded", return_value=False),
            patch.object(ft, "is_gil_disabled_build", return_value=True),
        ):
            assert "GIL is currently ENABLED" in ft.free_threading_status()
        with (
            patch.object(ft, "is_free_threaded", return_value=False),
            patch.object(ft, "is_gil_disabled_build", return_value=False),
        ):
            assert "standard (GIL) build" in ft.free_threading_status()


# --------------------------------------------------------------------------- #
# _run_free_threaded_server wiring
# --------------------------------------------------------------------------- #
@pytest.mark.xdist_group("proxy_cli")
class TestRunFreeThreadedServer:
    def _mocks(self, num_workers):
        servers = [MagicMock(name=f"server{i}") for i in range(num_workers)]
        server_cls = MagicMock(side_effect=servers)
        config_cls = MagicMock()

        listen_sock = MagicMock(name="listen_sock")
        dup_socks = [MagicMock(name=f"dup{i}") for i in range(num_workers)]
        listen_sock.dup.side_effect = dup_socks

        thread_objs = []

        def make_thread(target=None, args=(), name=None, daemon=None):
            t = MagicMock(name=name)
            t.is_alive.return_value = False
            t._target = target
            t._args = args
            t._daemon = daemon
            thread_objs.append(t)
            return t

        captured_signals = {}

        def fake_signal(sig, handler):
            captured_signals[sig] = handler

        return (
            servers,
            server_cls,
            config_cls,
            listen_sock,
            dup_socks,
            thread_objs,
            make_thread,
            captured_signals,
        )

    def test_spawns_one_server_and_socket_per_worker(self):
        num_workers = 4
        (
            servers,
            server_cls,
            config_cls,
            listen_sock,
            dup_socks,
            thread_objs,
            make_thread,
            captured_signals,
        ) = self._mocks(num_workers)

        uvicorn_stub = MagicMock()
        uvicorn_stub.Server = server_cls
        uvicorn_stub.Config = config_cls

        with (
            patch.dict(sys.modules, {"uvicorn": uvicorn_stub}),
            patch("socket.socket", return_value=listen_sock),
            patch("threading.Thread", side_effect=make_thread),
            patch("signal.signal", side_effect=fake_signal_factory(captured_signals)),
        ):
            ProxyInitializationHelpers._run_free_threaded_server(
                host="127.0.0.1",
                port=4000,
                num_workers=num_workers,
                uvicorn_args={
                    "app": "litellm.proxy.proxy_server:app",
                    "host": "127.0.0.1",
                    "port": 4000,
                    "workers": 99,  # must be stripped before Config()
                },
            )

        # One Config + Server per worker
        assert config_cls.call_count == num_workers
        assert server_cls.call_count == num_workers
        # "workers" kwarg must not leak into uvicorn.Config
        for call in config_cls.call_args_list:
            assert "workers" not in call.kwargs

        # Socket bound once, then dup()'d once per worker
        listen_sock.bind.assert_called_once_with(("127.0.0.1", 4000))
        listen_sock.listen.assert_called_once()
        assert listen_sock.dup.call_count == num_workers

        # One daemon thread per worker, each handed its own dup socket
        assert len(thread_objs) == num_workers
        for t in thread_objs:
            assert t._daemon is True
            t.start.assert_called_once()

        # Clean shutdown: every server told to exit, listen socket closed
        for s in servers:
            assert s.should_exit is True
        listen_sock.close.assert_called_once()

    def test_signal_handler_sets_should_exit(self):
        import signal as _signal

        num_workers = 2
        (
            servers,
            server_cls,
            config_cls,
            listen_sock,
            dup_socks,
            thread_objs,
            make_thread,
            captured_signals,
        ) = self._mocks(num_workers)

        uvicorn_stub = MagicMock()
        uvicorn_stub.Server = server_cls
        uvicorn_stub.Config = config_cls

        with (
            patch.dict(sys.modules, {"uvicorn": uvicorn_stub}),
            patch("socket.socket", return_value=listen_sock),
            patch("threading.Thread", side_effect=make_thread),
            patch("signal.signal", side_effect=fake_signal_factory(captured_signals)),
        ):
            ProxyInitializationHelpers._run_free_threaded_server(
                host="0.0.0.0",
                port=4000,
                num_workers=num_workers,
                uvicorn_args={"app": "x:app", "host": "0.0.0.0", "port": 4000},
            )

        assert _signal.SIGINT in captured_signals
        assert _signal.SIGTERM in captured_signals
        # Reset and re-trigger via the captured handler
        for s in servers:
            s.should_exit = False
        captured_signals[_signal.SIGINT](_signal.SIGINT, None)
        for s in servers:
            assert s.should_exit is True


def fake_signal_factory(store):
    def fake_signal(sig, handler):
        store[sig] = handler

    return fake_signal


# --------------------------------------------------------------------------- #
# benchmark metric helpers
# --------------------------------------------------------------------------- #
def _load_benchmark_module():
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "load_tests",
        "free_threading_benchmark",
        "benchmark.py",
    )
    spec = importlib.util.spec_from_file_location(
        "_ft_benchmark", os.path.abspath(path)
    )
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses + `from __future__ import annotations`
    # resolve field types via sys.modules[cls.__module__].
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestBenchmarkHelpers:
    @classmethod
    def setup_class(cls):
        cls.bm = _load_benchmark_module()

    def test_percentile(self):
        vals = [float(x) for x in range(1, 101)]  # 1..100
        assert self.bm._percentile(vals, 0.50) == pytest.approx(50.5)
        assert self.bm._percentile(vals, 0.90) == pytest.approx(90.1)
        assert self.bm._percentile([], 0.9) == 0.0

    def test_count_tokens(self):
        assert self.bm._count_tokens("a b c d") == 4
        assert self.bm._count_tokens("") == 0

    def test_extract_stream_text_openai_and_anthropic(self):
        openai_chunk = {"choices": [{"delta": {"content": "hello"}}]}
        assert self.bm._extract_stream_text(openai_chunk) == "hello"

        anthropic_delta = {
            "type": "content_block_delta",
            "delta": {"text": "world"},
        }
        assert self.bm._extract_stream_text(anthropic_delta) == "world"

        anthropic_full = {"content": [{"type": "text", "text": "full body"}]}
        assert self.bm._extract_stream_text(anthropic_full) == "full body"

    def test_extract_usage_tokens(self):
        assert self.bm._extract_usage_tokens({"usage": {"completion_tokens": 12}}) == 12
        assert self.bm._extract_usage_tokens({"usage": {"output_tokens": 7}}) == 7
        assert self.bm._extract_usage_tokens({}) == 0

    def test_phase_result_rps_and_tpm(self):
        res = self.bm.PhaseResult(
            label="x", streaming=False, duration_s=10.0, completed=100
        )
        assert res.rps == pytest.approx(10.0)
        res2 = self.bm.PhaseResult(
            label="y", streaming=True, duration_s=60.0, total_tokens=600
        )
        assert res2.tpm == pytest.approx(600.0)

    def test_payload_shapes(self):
        chat_stream = self.bm._payload("chat", True)
        assert chat_stream["stream"] is True
        assert chat_stream["stream_options"] == {"include_usage": True}
        msgs = self.bm._payload("messages", False)
        assert "stream" not in msgs
        assert msgs["model"] == "bench-messages"
