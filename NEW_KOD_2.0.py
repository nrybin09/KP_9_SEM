# gps_ephemeris.py
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import re

# Парсер чисел в стиле RINEX (поддержка E/D-экспонент)
_FLOAT_RE = re.compile(r'[+\-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+\-]?\d+)?')

def _to_floats(s: str) -> List[float]:
    s = s.replace('D', 'E').replace('d', 'E')
    return [float(m.group(0)) for m in _FLOAT_RE.finditer(s)]

def _read_body(fp: str) -> List[str]:
    lines = Path(fp).read_text(encoding="utf-8", errors="ignore").splitlines()
    # пропускаем заголовок до 'END OF HEADER'
    for i, ln in enumerate(lines[:2000]):
        if 'END OF HEADER' in ln:
            return lines[i+1:]
    return lines
@dataclass
class EphRec:  # GPS
    nSat: int = 0
    prn: int = 0
    af2: float = 0.0
    af1: float = 0.0
    af0: float = 0.0
    m0: float = 0.0
    sqrtA: float = 0.0
    deltan: float = 0.0
    e: float = 0.0
    omega: float = 0.0
    cuc: float = 0.0
    cus: float = 0.0
    crc: float = 0.0
    crs: float = 0.0
    i0: float = 0.0
    iDot: float = 0.0
    cic: float = 0.0
    cis: float = 0.0
    omega0: float = 0.0
    omegaDot: float = 0.0
    toe: float = 0.0
    toc: float = 0.0
    gpsWeekNo: int = 0
    svaccur: float = 0.0
    svhealth: float = 0.0
    iode: float = 0.0
    tgd: float = 0.0
    freqNo: int = 0     # для совместимости интерфейса
    debugId: int = 0

@dataclass
class GloRec:  # GLONASS
    nSat: int = 0
    prn: int = 0         # R-номер (1..24)
    freqNo: int = 0      # номер частотного канала (−7..+6)
    debugId: int = 0

    # часы и время кадра
    tauN: float = 0.0    # clock bias, с
    gammaN: float = 0.0  # относительная поправка частоты
    tk: float = 0.0      # сек от начала суток

    # орбита (коорд/скорость/ускорение)
    x: float = 0.0; dx: float = 0.0; ddx: float = 0.0
    y: float = 0.0; dy: float = 0.0; ddy: float = 0.0
    z: float = 0.0; dz: float = 0.0; ddz: float = 0.0

    svhealth: float = 0.0
    age: float = 0.0
def _parse_epoch_header_gps(line: str) -> Tuple[int, datetime, List[float]]:
    """
    Заголовок строки GPS: 'PRN yyyy mm dd hh mm ss' + 'af0 af1 af2' в хвосте.
    В некоторых файлах год может быть 'yy', адаптируй при необходимости.
    """
    t = line.rstrip('\n')
    prn = int(t[0:2])
    year   = int(t[2:6])
    month  = int(t[6:9])
    day    = int(t[9:12])
    hour   = int(t[12:15])
    minute = int(t[15:18])
    second = int(float(t[18:22]))
    afs = _to_floats(t[22:])
    dt = datetime(year, month, day, hour, minute, int(second))
    return prn, dt, afs[:3]  # af0, af1, af2

def rinex_get_nav_gps(file_nav: str) -> List[List[float]]:
    """
    Возвращает матрицу eRdy (29 x N) в «классической» раскладке:
      1:prn 2:af2 3:M0 4:sqrtA 5:deltan 6:e 7:omega 8:cuc 9:cus 10:crc
      11:crs 12:i0 13:idot 14:cic 15:cis 16:Omega0 17:Omegadot 18:toe
      19:af0 20:af1 21:toc 22:IODE 23:code_on_L2 24:weekno 25:L2flag
      26:svaccur 27:svhealth 28:tgd 29:fit_int
    """
    body = _read_body(file_nav)
    cols: List[List[float]] = []
    i = 0
    while i < len(body):
        if not body[i].strip():
            i += 1
            continue
        if i + 7 >= len(body):
            break

        lin1 = body[i];   lin2 = body[i+1]; lin3 = body[i+2]; lin4 = body[i+3]
        lin5 = body[i+4]; lin6 = body[i+5]; lin7 = body[i+6]; lin8 = body[i+7]
        i += 8

        # пропускаем не-GPS (например, Rxx)
        if not lin1[:2].strip().isdigit():
            continue

        prn, epoch_dt, af = _parse_epoch_header_gps(lin1)
        af0, af1, af2 = (af + [0.0, 0.0, 0.0])[:3]

        l2 = _to_floats(lin2); l3 = _to_floats(lin3); l4 = _to_floats(lin4)
        l5 = _to_floats(lin5); l6 = _to_floats(lin6); l7 = _to_floats(lin7)
        l8 = _to_floats(lin8)

        weekno = int(l6[2]) if len(l6) > 2 else 0
        hour = int(lin1[12:15]); minute = int(lin1[15:18]); second = int(float(lin1[18:22]))
        toc = hour*3600 + minute*60 + second

        col = [0.0]*29
        col[0]  = float(prn)
        col[1]  = af2
        col[2]  = (l2[3] if len(l2)>3 else 0.0)  # M0
        col[3]  = (l3[3] if len(l3)>3 else 0.0)  # sqrtA
        col[4]  = (l2[2] if len(l2)>2 else 0.0)  # deltan
        col[5]  = (l3[1] if len(l3)>1 else 0.0)  # e
        col[6]  = (l5[2] if len(l5)>2 else 0.0)  # omega
        col[7]  = (l3[0] if len(l3)>0 else 0.0)  # cuc
        col[8]  = (l3[2] if len(l3)>2 else 0.0)  # cus
        col[9]  = (l5[1] if len(l5)>1 else 0.0)  # crc
        col[10] = (l2[1] if len(l2)>1 else 0.0)  # crs
        col[11] = (l5[0] if len(l5)>0 else 0.0)  # i0
        col[12] = (l6[0] if len(l6)>0 else 0.0)  # idot
        col[13] = (l4[1] if len(l4)>1 else 0.0)  # cic
        col[14] = (l4[3] if len(l4)>3 else 0.0)  # cis
        col[15] = (l4[2] if len(l4)>2 else 0.0)  # Omega0
        col[16] = (l5[3] if len(l5)>3 else 0.0)  # Omegadot
        col[17] = (l4[0] if len(l4)>0 else 0.0)  # toe
        col[18] = af0
        col[19] = af1
        col[20] = float(toc)
        col[21] = (l2[0] if len(l2)>0 else 0.0)  # IODE
        col[22] = (l6[1] if len(l6)>1 else 0.0)  # code_on_L2
        col[23] = float(weekno)
        col[24] = (l6[3] if len(l6)>3 else 0.0)  # L2flag
        col[25] = (l7[0] if len(l7)>0 else 0.0)  # svaccur
        col[26] = (l7[1] if len(l7)>1 else 0.0)  # svhealth
        col[27] = (l7[2] if len(l7)>2 else 0.0)  # tgd
        col[28] = (l8[1] if len(l8)>1 else 0.0)  # fit_int

        cols.append(col)

    if not cols:
        return []
    n = len(cols)
    eRdy = [[cols[j][i] for j in range(n)] for i in range(29)]  # 29 x N
    return eRdy


def rinex_get_nav_glonass(file_nav: str) -> List[Dict]:
    """
    Читает записи ГЛОНАСС (Rxx) из RINEX NAV/BRDC.
    Возвращает список словарей:
      {prn, tauN, gammaN, tk, freqNo, x,dx,ddx, y,dy,ddy, z,dz,ddz, svhealth, age}
    """
    body = _read_body(file_nav)
    out: List[Dict] = []
    i = 0
    while i < len(body):
        line = body[i]
        if not line.strip():
            i += 1
            continue
        if not (len(line) >= 1 and line[0] == 'R'):
            i += 1
            continue
        if i + 3 >= len(body):
            break

        l1 = body[i]; l2 = body[i+1]; l3 = body[i+2]; l4 = body[i+3]
        i += 4

        prn = int(l1[1:3])
        nums1 = _to_floats(l1[22:])
        tauN  = nums1[0] if len(nums1) > 0 else 0.0
        gammaN= nums1[1] if len(nums1) > 1 else 0.0
        tk    = nums1[2] if len(nums1) > 2 else 0.0

        a2 = _to_floats(l2); a3 = _to_floats(l3); a4 = _to_floats(l4)
        x, dx, ddx = (a2 + [0,0,0])[:3]
        y, dy, ddy = (a3 + [0,0,0])[:3]
        z, dz, ddz = (a4 + [0,0,0])[:3]

        svhealth = 0.0; age = 0.0; freqNo = 0
        tail = (a2[3:] if len(a2) > 3 else []) + (a3[3:] if len(a3) > 3 else []) + (a4[3:] if len(a4) > 3 else [])
        if tail:
            svhealth = float(tail[-1])
        if len(tail) >= 2:
            age = float(tail[-2])
        if len(tail) >= 3:
            freqNo = int(round(tail[-3]))

        out.append(dict(
            prn=prn, tauN=tauN, gammaN=gammaN, tk=tk, freqNo=freqNo,
            x=x, dx=dx, ddx=ddx, y=y, dy=dy, ddy=ddy, z=z, dz=dz, ddz=ddz,
            svhealth=svhealth, age=age
        ))
    return out
def read_gps_ephemeris(file_nav: str):
    """
    Возвращает:
      eph      — [k][j] элементы типа EphRec (j=PRN)
      ephSats  — список по k: PRN, доступные в этот toe
      ephT     — список [toe, gpsWeekNo]
    """
    eRdy = rinex_get_nav_gps(file_nav)  # 29 x N
    if not eRdy:
        return [], [], []

    toes = sorted({eRdy[17][c] for c in range(len(eRdy[0]))})  # toe
    ephT = [[toe, int(eRdy[23][0]) if eRdy[23] else 0] for toe in toes]

    K = len(toes)
    eph: List[List[Optional[EphRec]]] = [[None for _ in range(33)] for _ in range(K)]
    ephSats: List[List[int]] = [[] for _ in range(K)]

    toe_to_cols: Dict[float, List[int]] = {}
    for col in range(len(eRdy[0])):
        toe = eRdy[17][col]
        toe_to_cols.setdefault(toe, []).append(col)

    for k, toe in enumerate(toes):
        cols = toe_to_cols[toe]
        nSat = len(cols)
        sats: List[int] = []
        for n in cols:
            j = int(eRdy[0][n])  # PRN
            sats.append(j)
            rec = EphRec()
            rec.nSat = nSat; rec.prn = j
            rec.af2  = eRdy[1][n]
            rec.af1  = eRdy[19][n]
            rec.af0  = eRdy[18][n]
            rec.m0   = eRdy[2][n]
            rec.sqrtA= eRdy[3][n]
            rec.deltan = eRdy[4][n]
            rec.e      = eRdy[5][n]
            rec.omega  = eRdy[6][n]
            rec.cuc    = eRdy[7][n]
            rec.cus    = eRdy[8][n]
            rec.crc    = eRdy[9][n]
            rec.crs    = eRdy[10][n]
            rec.i0     = eRdy[11][n]
            rec.iDot   = eRdy[12][n]
            rec.cic    = eRdy[13][n]
            rec.cis    = eRdy[14][n]
            rec.omega0 = eRdy[15][n]
            rec.omegaDot = eRdy[16][n]
            rec.toe    = eRdy[17][n]
            rec.toc    = eRdy[20][n]
            rec.gpsWeekNo = int(eRdy[23][n])
            rec.svaccur   = eRdy[25][n]
            rec.svhealth  = eRdy[26][n]
            rec.iode      = eRdy[21][n]
            rec.tgd       = eRdy[27][n]
            rec.freqNo    = 0
            rec.debugId   = j
            eph[k][j] = rec
        ephSats[k] = sats

    return eph, ephSats, ephT

def read_glonass_ephemeris(file_nav: str):
    """
    Возвращает:
      ephR     — [k][j] элементы типа GloRec (j=R-номер)
      ephRSats — список по k: PRN, доступные в этот tk
      ephRT    — список [tk, 0] (второй элемент — свободный слот)
    """
    rows = rinex_get_nav_glonass(file_nav)
    if not rows:
        return [], [], []

    tks = sorted({r['tk'] for r in rows})
    K = len(tks)
    ephR: List[List[Optional[GloRec]]] = [[None for _ in range(33)] for _ in range(K)]
    ephRSats: List[List[int]] = [[] for _ in range(K)]
    ephRT = [[tk, 0] for tk in tks]

    tk_to_rows: Dict[float, List[Dict]] = {}
    for r in rows:
        tk_to_rows.setdefault(r['tk'], []).append(r)

    for k, tk in enumerate(tks):
        group = tk_to_rows[tk]
        nSat = len(group)
        sats: List[int] = []
        for r in group:
            j = r['prn']
            sats.append(j)
            rec = GloRec()
            rec.nSat = nSat
            rec.prn = j
            rec.freqNo = int(r['freqNo'])
            rec.debugId = j
            rec.tauN = r['tauN']; rec.gammaN = r['gammaN']; rec.tk = r['tk']
            rec.x, rec.dx, rec.ddx = r['x'], r['dx'], r['ddx']
            rec.y, rec.dy, rec.ddy = r['y'], r['dy'], r['ddy']
            rec.z, rec.dz, rec.ddz = r['z'], r['dz'], r['ddz']
            rec.svhealth = r['svhealth']; rec.age = r['age']
            if j >= len(ephR[k]):
                ephR[k].extend([None] * (j - len(ephR[k]) + 1))
            ephR[k][j] = rec
        ephRSats[k] = sats

    return ephR, ephRSats, ephRT
class EphAccessor:
    """ Доступ как в MATLAB: eph[k][j].sqrtA, ephR[k][j].freqNo и т.п. """
    def __init__(self, eph_matrix):
        self._eph = eph_matrix
    def __getitem__(self, k: int):
        return self._eph[k]

def load_nav_separate(gps_path: str, glo_path: str):
    """
    Читает эфемериды ИЗ ДВУХ ФАЙЛОВ:
      gps_path — файл с GPS-записями (PRN как числа в заголовке)
      glo_path — файл с GLONASS-записями ('R' в начале блока)
    Возвращает:
      (gps, gpsSats, gpsT), (glo, gloSats, gloT)
    где gps/glo — EphAccessor (обёртка).
    """
    ephG, ephGSats, ephGT = read_gps_ephemeris(gps_path)
    ephR, ephRSats, ephRT = read_glonass_ephemeris(glo_path)
    return (EphAccessor(ephG), ephGSats, ephGT), (EphAccessor(ephR), ephRSats, ephRT)

if __name__ == "__main__":
    # Пример локального теста — укажи свои файлы:
    GPS_FILE = "BRDC0010.17n"
    GLO_FILE = "base_01.rnx"
    if Path(GPS_FILE).exists() and Path(GLO_FILE).exists():
        (gps, gpsSats, gpsT), (glo, gloSats, gloT) = load_nav_separate(GPS_FILE, GLO_FILE)

# --- вывод всех доступных спутников по эпохам ---
print("\n== Список спутников по эпохам: GPS ==")
for kk, prns in enumerate(gpsSats):
    toe = gpsT[kk][0] if kk < len(gpsT) else None
    print(f"k={kk:2d}, toe={toe:>10.0f}  -> PRN: {sorted(prns)}")

print("\n== Список спутников по эпохам: GLONASS ==")
for kk, prns in enumerate(gloSats):
    tk = gloT[kk][0] if kk < len(gloT) else None
    print(f"k={kk:2d}, tk={tk:>10.0f}  -> R: {sorted(prns)}")

# --- теперь задаём индексы прямо в коде ---
k = 0   # номер эпохи (можно менять)
j = 3   # номер спутника (PRN для GPS или R для ГЛОНАСС)

print(f"\nВыбрана эпоха k={k}, спутник j={j}")

# --- автоматическое определение, GPS или ГЛОНАСС ---
if k < len(gpsSats) and j in gpsSats[k] and gps[k][j] is not None:
    rec = gps[k][j]
    print(f"\n[GPS] PRN{j}:")
    print(f"sqrtA = {rec.sqrtA}")
    print(f"e      = {rec.e}")
    print(f"toe    = {rec.toe}")
elif k < len(gloSats) and j in gloSats[k] and glo[k][j] is not None:
    rec = glo[k][j]
    print(f"\n[GLONASS] R{j}:")
    print(f"x       = {rec.x}")
    print(f"y       = {rec.y}")
    print(f"z       = {rec.z}")
    print(f"freqNo  = {rec.freqNo}")
    print(f"tk      = {rec.tk}")
else:
    print("\nТакого спутника нет в выбранной эпохе.")

# # Для GPS
# print(f"Все спутники для эпохи {k} в GPS:")
# for j in gpsSats[k]:
#     print(f"PRN {j}: sqrtA = {gps[k][j].sqrtA}, e = {gps[k][j].e}, toe = {gps[k][j].toe}")

# # Для ГЛОНАСС
# print(f"Все спутники для эпохи {k} в ГЛОНАСС:")
# for j in gloSats[k]:
#     print(f"R{j}: x = {glo[k][j].x}, y = {glo[k][j].y}, z = {glo[k][j].z}, freqNo = {glo[k][j].freqNo}")
