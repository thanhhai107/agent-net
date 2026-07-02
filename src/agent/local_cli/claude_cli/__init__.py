"""LangGraph + Claude Code CLI agents.

Same two-phase orchestration as ``agent.byo.langgraph.react_agent.BasicReActAgent``,
but worker nodes invoke ``claude -p`` subprocesses instead of LangChain
``create_agent`` graphs.  Model defaults and authentication are handled by
:mod:`agent.local_cli.claude_cli.config`.

Layout::

    local_cli/claude_cli/
      agent.py                    # ClaudeAgent — StateGraph orchestrator
      config.py                   # Model defaults and auth helpers
      claude_worker.py            # ClaudeWorker — subprocess adapter
      claude_display.py           # Claude stream-json event formatter
      phases/
        diagnosis.py              # ClaudeDiagnosisPhase
        submission.py             # ClaudeSubmissionPhase
"""

from agent.local_cli.claude_cli.agent import ClaudeAgent

__all__ = ["ClaudeAgent"]
