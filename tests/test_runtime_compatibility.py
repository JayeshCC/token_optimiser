"""Exercise the actual HTTP/WebSocket runtime without provider credentials."""
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from token_optimiser import TokenOptimiserAction, TokenOptimiserEnv


@pytest.fixture(params=[False, True], ids=['api', 'native-web-ui'])
def live_server(request, tmp_path):
    with socket.socket() as port_socket:
        port_socket.bind(('127.0.0.1', 0))
        port = port_socket.getsockname()[1]
    env = os.environ.copy()
    for key in ['HF_TOKEN', 'API_KEY', 'OPENAI_API_KEY']:
        env.pop(key, None)
    env['HF_HOME'] = str(tmp_path/'empty-hf-home')
    env['ENABLE_WEB_INTERFACE'] = str(request.param).lower()
    root = Path(__file__).resolve().parents[1]
    base = f'http://127.0.0.1:{port}'
    with (tmp_path/'server.log').open('w+') as log:
        process = subprocess.Popen(
            [sys.executable, '-m', 'uvicorn', 'server.app:app', '--host', '127.0.0.1',
             '--port', str(port), '--lifespan', 'on'],
            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic()+20
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(base+'/health', timeout=1) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                if process.poll() is not None:
                    break
                time.sleep(0.1)
            else:
                pytest.fail('Normal application startup timed out')
            if process.poll() is not None:
                log.seek(0)
                pytest.fail('Server exited during startup: '+log.read())
            yield base, request.param
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_http_health_schema_and_optional_ui(live_server):
    base, web_enabled = live_server
    with urllib.request.urlopen(base+'/health', timeout=3) as response:
        assert response.status == 200
    with urllib.request.urlopen(base+'/schema', timeout=3) as response:
        schema = json.load(response)
    assert 'optimized_prompt' in json.dumps(schema)
    if web_enabled:
        with urllib.request.urlopen(base+'/web', timeout=3) as response:
            assert response.status == 200
            assert 'text/html' in response.headers['Content-Type']


def test_typed_client_reset_step_and_state(live_server):
    base, _ = live_server
    with TokenOptimiserEnv(base_url=base) as client:
        initial = client.reset()
        assert not initial.done
        assert client.state().step_count == 0
        result = client.step(TokenOptimiserAction(optimized_prompt='Explain photosynthesis briefly.'))
        assert result.observation.llm_response
        assert 0 <= result.reward <= 1
        assert client.state().step_count == 1


def test_concurrent_clients_keep_independent_state(live_server):
    base, _ = live_server
    with TokenOptimiserEnv(base_url=base) as first, TokenOptimiserEnv(base_url=base) as second:
        first.reset()
        second.reset()
        first.step(TokenOptimiserAction(optimized_prompt='Define photosynthesis.'))
        assert first.state().step_count == 1
        assert second.state().step_count == 0
        assert first.state().episode_id != second.state().episode_id


def test_invalid_action_does_not_destroy_session(live_server):
    from websockets.sync.client import connect
    base, _ = live_server
    with connect(base.replace('http://', 'ws://')+'/ws', open_timeout=3) as connection:
        connection.send(json.dumps({'type': 'reset'}))
        assert json.loads(connection.recv(timeout=3))['type'] == 'observation'
        connection.send(json.dumps({'type': 'step', 'data': {'action': {}}}))
        assert json.loads(connection.recv(timeout=3))['type'] == 'error'
        connection.send(json.dumps({'type': 'state'}))
        assert json.loads(connection.recv(timeout=3))['type'] == 'state'


def test_inference_preserves_provider_free_baseline(live_server, tmp_path):
    base, _ = live_server
    env = os.environ.copy()
    for key in ['HF_TOKEN', 'API_KEY', 'OPENAI_API_KEY', 'LOCAL_IMAGE_NAME']:
        env.pop(key, None)
    env.update(SERVER_URL=base, HF_HOME=str(tmp_path/'empty-hf-home'), HF_HUB_OFFLINE='1')
    result = subprocess.run(
        [sys.executable, 'inference.py'], cwd=Path(__file__).resolve().parents[1],
        env=env, capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stderr
    # Independently reproduced with the unchanged original 0.3.0 runtime.
    assert '[END] success=false steps=9 score=0.430' in result.stdout, result.stdout+result.stderr


def test_runtime_has_no_image_ui_dependency_chain():
    # A runtime upgrade must be reviewed before reintroducing these dependencies.
    for name in ['gradio', 'pillow']:
        with pytest.raises(importlib.metadata.PackageNotFoundError):
            importlib.metadata.version(name)
