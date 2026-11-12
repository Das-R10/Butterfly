import os
import sys

_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
_ba_dir = os.path.join(_root, "ba")
if _ba_dir not in sys.path:
    sys.path.insert(0, _ba_dir)

_target_src = os.path.join(_ba_dir, "src")
if _target_src not in __path__:
    __path__.append(_target_src)

