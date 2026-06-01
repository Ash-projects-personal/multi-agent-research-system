"""
pytest configuration for multi-agent-research-system tests.

Makes the repo root importable so `from orchestrator import AgentOrchestrator`
works without an installed package.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
