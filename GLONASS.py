from __future__ import annotations
from typing import List, Dict
import re

# ловим не 0 начение 
_FLOAT_RE = re.compile(r'[+\-]?(?:\d+\.\d*|\.\d+)(?:[EeDd][+\-]?\d+)?')

# порядок имён строго соответствует порядку чисел в блоке RINEX ГЛОНАСС
# 1-я строка: tauN, gammaN, tk
# 2-я строка: x, vx, ax, health
# 3-я строка: y, vy, ay, freq_num
# 4-я строка: z, vz, az, age
_FIELD_ORDER = [
    "tauN","gammaN","tk",
    "x","vx","ax","health",
    "y","vy","ay","freq_num",
    "z","vz","az","age"
]

def _floats_flat(block_lines: List[str]) -> List[float]:
    """Достаём ВСЕ числа по порядку из блока (заменяя D→E)."""
    out: List[float] = []
    for ln in block_lines:
        s = ln.replace('D', 'E').replace('d', 'e')
        out.extend(float(m.group(0)) for m in _FLOAT_RE.finditer(s))
    return out

def parse_glo_block(block_lines: List[str]) -> Dict[str, float]:
    """
    Возвращает словарь из 15 параметров ГЛОНАСС в том же порядке,
    что и в «сыром блоке». Первое служебное число (обычно 0.0) пропускаем.
    """
    nums = _floats_flat(block_lines)

    # Если в начале стоит 0.0 (служебная метка) — пропускаем её
    if nums and abs(nums[0]) < 1e-12:
        nums = nums[1:]

    # Берём первые 15 значений подряд
    vals: Dict[str, float] = {}
    for i, name in enumerate(_FIELD_ORDER):
        vals[name] = nums[i] if i < len(nums) else float("nan")

    return vals
