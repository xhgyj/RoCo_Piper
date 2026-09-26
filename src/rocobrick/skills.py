"""Compatibility source for the skills package.

The runtime imports :mod:`rocobrick.skills` as a package.  The implementation
lives in ``skills/programs.py``; this file remains for static compatibility.
"""
from rocobrick.compat import StrEnum
from rocobrick.skills.programs import *
