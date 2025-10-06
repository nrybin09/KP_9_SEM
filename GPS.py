from __future__ import annotations
from typing import List, Dict
import re

# ловим не 0 начение 
_FLOAT_RE = re.compile(r'[+\-]?(?:\d+\.\d*|\.\d+)(?:[EeDd][+\-]?\d+)?')

# порядок имён строго соответствует порядку чисел в блоке RINEX GPS
# (после строки с PRN и эпохой спутника)
#
# 1-я строка (конец строки с эпохой): af0, af1, af2
# 2-я строка: IODE, Crs, dn, M0
# 3-я строка: Cuc, e, Cus, sqrtA
# 4-я строка: Toe, Cic, OMEGA0, Cis
# 5-я строка: i0, Crc, omega, OMEGADOT
# 6-я строка: IDOT, L2_codes, week, L2P_flag
# 7-я строка: SVacc, SVhealth, TGD, IODC
# 8-я строка: t_trans, fit_int

_FIELD_ORDER = [
    "af0","af1","af2",
    "IODE","Crs","dn","M0",
    "Cuc","e","Cus","sqrtA",
    "Toe","Cic","OMEGA0","Cis",
    "i0","Crc","omega","OMEGADOT",
    "IDOT","L2_codes","week","L2P_flag",
    "SVacc","SVhealth","TGD","IODC",
    "t_trans","fit_int"
]

def _floats_flat(block_lines: List[str]) -> List[float]:
    """Достаём ВСЕ числа по порядку из блока (заменяя D→E)."""
    out: List[float] = []
    for ln in block_lines:
        s = ln.replace('D', 'E').replace('d', 'e')
        out.extend(float(m.group(0)) for m in _FLOAT_RE.finditer(s))
    return out

def parse_gps_block(block_lines: List[str]) -> Dict[str, float]:
    """
    Возвращает словарь из 29 параметров GPS в том же порядке,
    что и в «сыром блоке». Первое служебное число (обычно 0.0) пропускаем.
    """
    nums = _floats_flat(block_lines)

    # Если в начале стоит 0.0 (служебная метка) — пропускаем её
    if nums and abs(nums[0]) < 1e-12:
        nums = nums[1:]

    # Берём первые значения подряд
    vals: Dict[str, float] = {}
    for i, name in enumerate(_FIELD_ORDER):
        vals[name] = nums[i] if i < len(nums) else float("nan")

    return vals
