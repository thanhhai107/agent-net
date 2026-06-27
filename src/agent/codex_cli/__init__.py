"""LangGraph + Codex CLI agents.

Same two-phase orchestration as ``agent.langgraph.react_agent.BasicReActAgent``,
but worker nodes invoke ``codex exec`` subprocesses instead of LangChain
``create_agent`` graphs.

Layout::

    codex_cli/
      agent.py                    # CodexCliAgent — StateGraph orchestrator
      codex_worker.py             # CodexWorker — subprocess adapter
      phases/
        diagnosis.py              # CodexCliDiagnosisPhase
        submission.py             # CodexCliSubmissionPhase
"""

from agent.codex_cli.agent import CodexCliAgent

__all__ = ["CodexCliAgent"]
