# Dependency security status

The lockfile was refreshed on 28 October 2026 without rewriting history or changing project ownership. Package/model imports, environment construction/reset in deterministic fallback mode, and the direct health endpoint were checked. A bounded HTTP health smoke also passed with ASGI lifespan disabled; normal application startup/lifespan remains unverified. No provider requests or paid API calls were performed. Full provider workflows and a production deployment are not certified.

## Upstream constraint

OpenEnv core currently depends on Gradio. The resolved Gradio 6.31.0 requires `pillow>=8,<13`, so the lock resolves Pillow 12.3.0. GitHub advisories affecting versions below Pillow 13 therefore remain unresolved. Forcing Pillow 13 would violate that upstream constraint and is not presented as a validated fix.

Do not expose this historical prototype to untrusted files or a public network. A compatible upstream release, or a separately tested replacement of the UI dependency path, is needed before claiming those advisories are resolved. No alerts were dismissed to conceal this constraint.

## Python compatibility

The minimum Python version is now 3.11. Python 3.10 caused the universal lock to retain vulnerable older dnspython and filelock releases; the refreshed lock no longer retains those compatibility branches. Local verification used Python 3.13. Dropping Python 3.10 support is deliberate and does not resolve the separate Pillow/Gradio constraint above.
