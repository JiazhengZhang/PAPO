import os
import json
import datetime
from typing import List, Dict, Any, Optional, Callable, Iterator, Union
from pathlib import Path
from dataclasses import is_dataclass, asdict
from datetime import date, datetime
from enum import Enum
import collections.abc as cabc

import pandas as pd
import numpy as np

def json_sanitize(obj: Any, *, on_unknown: str = "drop", max_depth: int = 64) -> Any | None:
    _SKIP = object()

    def _inner(o: Any, depth: int) -> Any:
        if depth > max_depth:
            return _SKIP

        # 基础可序列化类型
        if isinstance(o, (str, int, float, bool)) or o is None:
            return o

        # 常见转换
        if isinstance(o, (bytes, bytearray)):
            return o.decode("utf-8", errors="replace")
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, Enum):
            return _inner(o.value, depth + 1)

        # dataclass
        if is_dataclass(o):
            o = asdict(o)

        # torch/numpy/pandas 等：duck-typing 转基础类型
        if hasattr(o, "detach") and callable(getattr(o, "detach")) and hasattr(o, "cpu"):
            try:
                return _inner(o.detach().cpu(), depth + 1)
            except Exception:
                pass
        if hasattr(o, "tolist"):
            try:
                return _inner(o.tolist(), depth + 1)
            except Exception:
                pass
        if hasattr(o, "item"):
            try:
                return _inner(o.item(), depth + 1)
            except Exception:
                pass

        # 映射类型
        if isinstance(o, cabc.Mapping):
            out = {}
            for k, v in o.items():
                try:
                    sk = k if isinstance(k, str) else str(k)
                except Exception:
                    continue
                sv = _inner(v, depth + 1)
                if sv is not _SKIP:
                    out[sk] = sv
            return out

        # 序列类型（包含 set / tuple / list）
        if isinstance(o, (list, tuple, set, frozenset)) or (isinstance(o, cabc.Sequence) and not isinstance(o, (str, bytes, bytearray))):
            out = []
            for it in list(o):  # set/frozenset 转列表
                si = _inner(it, depth + 1)
                if si is not _SKIP:
                    out.append(si)
            return out

        # 其他未知对象
        if on_unknown == "stringify":
            try:
                return str(o)
            except Exception:
                return _SKIP
        return _SKIP

    cleaned = _inner(obj, 0)
    return None if cleaned is _SKIP else cleaned