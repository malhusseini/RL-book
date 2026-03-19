"""dp — Value iteration / policy iteration wrappers and policy diagnostics."""

from .solver import solve, solve_and_plot, policy_summary

__all__ = ["solve", "solve_and_plot", "policy_summary"]
