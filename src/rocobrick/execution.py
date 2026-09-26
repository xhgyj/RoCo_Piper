"""Compatibility source for the execution package.

The runtime imports :mod:`rocobrick.execution` as a package.  The implementation
lives in ``execution/core.py``; this file remains for source compatibility and
static tooling that expects the historical module path.
"""
from rocobrick.compat import StrEnum
from rocobrick.execution.core import *
