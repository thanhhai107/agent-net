"""SDK-backed agents (planned).

Direct integration with vendor SDKs, bypassing LangChain chat models:
- vendor SDK adapters under ``agent.sdk`` (planned)

Expected layout::

    sdk/
      vendor_sdk/       # SDK agent adapter (planned)

Both phases (diagnosis → submission) should still write to
``{session_dir}/messages.jsonl`` via ``AgentCallbackLogger`` or an SDK-specific
adapter with the same event schema.
"""

__all__: list[str] = []
