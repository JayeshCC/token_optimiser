# Dependency security status

Updated 29 October 2026. The earlier Gradio/Pillow blocker is addressed through a tested compatibility runtime, not an incompatible package override or alert dismissal.

## Dependency change

`openenv-core[core]==0.2.1` provides the project's required HTTP/WebSocket runtime without depending on Gradio or Pillow. The refreshed `uv.lock` removes both packages and their unused UI dependency chain. `uv pip check` reports a consistent environment; no dependencies are excluded or forced against their declared requirements.

The selected SDK is synchronous. `inference.py` calls it through `asyncio.to_thread` so the existing async inference driver remains usable. This runtime offers its older native optional web interface, rather than the newer Gradio UI. Newer OpenEnv async-SDK/Gradio-specific features are outside this compatibility configuration. Re-evaluate the pin when a supported runtime resolves patched image/UI dependencies.

Python 3.11 remains the minimum version. That avoids vulnerable older dnspython/filelock releases needed by Python 3.10. All changes preserve repository ownership and forward commit history.

## Verification

- Normal ASGI startup/lifespan, HTTP health and schema endpoints.
- Typed WebSocket reset, step and state; independent simultaneous client sessions.
- Invalid action handling without losing the session.
- Both API-only and optional native web-interface modes.
- Full provider-free inference: nine steps and aggregate 0.430 under both the unchanged original 0.3.0 runtime and the compatible 0.2.1 runtime. The old README's 0.487 figure was stale; the scoring/environment implementation was not altered.
- `pip-audit`: no known vulnerabilities among 85 audited installed dependencies at verification. The local unpublished application cannot be looked up on PyPI and was reported separately as skipped, not certified secure.
- Regression CI covers Python 3.11 and 3.13; exact run results are available in GitHub Actions.

No provider requests or paid API calls were performed. Docker builds, provider-backed training and production deployment are not certified. This remains a historical prototype; keep it local unless authentication, resource limits and deployment security are reviewed separately.

## Upstream reference

Gradio's Pillow-13 support was merged for its 7.0 development line in [gradio-app/gradio#14005](https://github.com/gradio-app/gradio/pull/14005). Stable Gradio 6.31.0 still declares Pillow below 13, which motivated selecting the UI-free runtime dependency path instead of violating that bound.
