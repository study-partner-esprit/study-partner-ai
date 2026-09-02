"""Evaluator agent package.

Kept intentionally light: `EvaluatorAgent` builds a LiteLLM-backed LLM shim
lazily. Import `agents.evaluator.agent` explicitly (or use the worker's lazy
construction) so lightweight consumers of `agents.evaluator.schemas` /
`agents.evaluator.prompts` stay dependency-free.
"""
