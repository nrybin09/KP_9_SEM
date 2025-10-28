from pathlib import Path
from datetime import datetime
import re
_FLOAT_RE = re.compile(r'[+\-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+\-]?\d+)?')

def _to_floats(s):
    s = s.replace('D', 'E').replace('d', 'E')
    return [float(m.group(0)) for m in _FLOAT_RE.finditer(s)]

def _read_body(fp: str):
    lines = Path(fp).read_text(encoding="utf-8", errors="ignore").splitlines()
    # Ищем строку, содержащую 'END OF HEADER' — это конец заголовка RINEX-файла.
    for i, ln in enumerate(lines[:2000]):  # ограничиваем поиск первыми 2000 строками на случай повреждённого файла
        if 'END OF HEADER' in ln:
            # Возвращаем все строки, идущие после заголовка
            return lines[i+1:]
    # Если в файле нет строки 'END OF HEADER', возвращаем весь файл как есть.
    return lines

class GPSRec:
    def __init__(self):
        # - служебные -
        self.nSat = 0          # число спутников в эпохе
        self.prn  = 0          # PRN
        # - Строка 1: PRN / EPOCH / SV CLK -
        self.af0 = 0.0         # смещение часов (s)
        self.af1 = 0.0         # дрейф (s/s)
        self.af2 = 0.0         # ускорение дрейфа (s/s^2)
        # - Строка 2: BROADCAST ORBIT - 1 -
        self.iode   = 0.0      # IODE
        self.crs    = 0.0      # Crs (m)
        self.deltan = 0.0      # Δn (rad/s)
        self.m0     = 0.0      # M0 (rad)
        # - Строка 3: BROADCAST ORBIT - 2 -
        self.cuc   = 0.0       # Cuc (rad)
        self.e     = 0.0       # e
        self.cus   = 0.0       # Cus (rad)
        self.sqrtA = 0.0       # sqrt(A) (m^0.5)
        # - Строка 4: BROADCAST ORBIT - 3 -
        self.toe     = 0.0     # Toe (s of GPS week)
        self.cic     = 0.0     # Cic (rad)
        self.omega0  = 0.0     # Ω0 (rad)
        self.cis     = 0.0     # Cis (rad)
        # - Строка 5: BROADCAST ORBIT - 4 -
        self.i0       = 0.0    # i0 (rad)
        self.crc      = 0.0    # Crc (m)
        self.omega    = 0.0    # ω (rad)
        self.omegaDot = 0.0    # Ω̇ (rad/s)
        # - Строка 6: BROADCAST ORBIT - 5 -
        self.iDot     = 0.0    # IDOT (rad/s)
        self.codeL2   = 0.0    # code on L2
        self.gpsWeekNo= 0      # GPS week (continuous)
        self.L2flag   = 0.0    # L2 P data flag
        # - Строка 7: BROADCAST ORBIT - 6 -
        self.svaccur  = 0.0    # SV accuracy (m)
        self.svhealth = 0.0    # SV health
        self.tgd      = 0.0    # TGD (s)
        self.iodc     = 0.0    # IODC
        # - Строка 8: BROADCAST ORBIT - 7 -
        self.txTime = 0.0      # transmission time (s of GPS week)
        self.fitInt = 0.0      # fit interval (hours)
        # - доп. поле из твоего кода -
        self.toc = 0.0         # время часов (сек от начала суток), используется в обработке
        self.freqNo = 0        # для совместимости (не используется для GPS)
        self.debugId = 0       # ID для отладки (копия PRN)

class GloRec:
    def __init__(self):
        # - служебное -
        self.nSat = 0        # количество спутников в эпохе
        self.prn  = 0        # R-номер (1…24)
        # - строка 1: RPRN / EPOCH / SV CLK -
        # τN, γN, tk
        self.tauN   = 0.0    # смещение шкалы времени спутника (с)
        self.gammaN = 0.0    # относительная поправка частоты
        self.tk     = 0.0    # время момента измерения (с от начала суток)
        # - строка 2: BROADCAST ORBIT 1 -
        # x, ẋ, ẍ, freqNo
        self.x   = 0.0       # координата X (м)
        self.dx  = 0.0       # скорость X (м/с)
        self.ddx = 0.0       # ускорение X (м/с²)
        self.freqNo = 0      # номер частотного канала (−7 … +6)
        # - строка 3: BROADCAST ORBIT 2 -
        # y, ẏ, ÿ, age
        self.y   = 0.0       # координата Y (м)
        self.dy  = 0.0       # скорость Y (м/с)
        self.ddy = 0.0       # ускорение Y (м/с²)
        self.age = 0.0       # возраст данных эфемерид
        # - строка 4: BROADCAST ORBIT 3 -
        # z, ż, żż, svhealth
        self.z   = 0.0       # координата Z (м)
        self.dz  = 0.0       # скорость Z (м/с)
        self.ddz = 0.0       # ускорение Z (м/с²)
        self.svhealth = 0.0  # флаг исправности (0 = исправен)
        # - дополнительное -
        self.debugId = 0     # ID для отладки (копия PRN)

def rinex_get_nav_gps(file_nav):
    """
    читаем и формирует столбцы в порядке, как в файле.
      0: PRN               1: af0             2: af1            3: af2
      4: IODE              5: Crs             6: deltan         7: M0
      8: Cuc               9: e               10: Cus           11: sqrtA
      12: Toe              13: Cic            14: Omega0        15: Cis
      16: i0               17: Crc            18: omega         19: Omegadot
      20: IDOT             21: code_on_L2     22: GPS_week      23: L2flag
      24: sv_accuracy      25: sv_health      26: TGD           27: IODC
      28: transmission_time (t_Tx)      29: fit_interval (hours)
    """
    body = _read_body(file_nav)
    cols = []
    i = 0
    while i < len(body):
        if not body[i].strip():
            i += 1
            continue
        if i + 7 >= len(body):
            break

        lin1 = body[i]
        lin2 = body[i+1]
        lin3 = body[i+2]
        lin4 = body[i+3]
        lin5 = body[i+4]
        lin6 = body[i+5]
        lin7 = body[i+6]
        lin8 = body[i+7]
        i += 8

        # пропускаем не-GPS (например, Rxx)
        if not lin1[:2].strip().isdigit():
            continue

        # --- Строка 1: PRN, af0, af1, af2 ---
        prn = int(lin1[0:2])               # первые 2 символа — номер спутника (PRN)
        L1 = _to_floats(lin1[22:])         # преобразуем числа начиная с позиции 22
        af0 = L1[0] if len(L1) > 0 else 0.0
        af1 = L1[1] if len(L1) > 1 else 0.0
        af2 = L1[2] if len(L1) > 2 else 0.0

        # Остальные строки конвертируем в числа
        L2 = _to_floats(lin2)  # IODE, Crs, Δn, M0
        L3 = _to_floats(lin3)  # Cuc, e, Cus, sqrtA
        L4 = _to_floats(lin4)  # Toe, Cic, Ω0, Cis
        L5 = _to_floats(lin5)  # i0, Crc, ω, Ω̇
        L6 = _to_floats(lin6)  # IDOT, code_on_L2, GPS_week, L2flag
        L7 = _to_floats(lin7)  # sv_accuracy, sv_health, TGD, IODC
        L8 = _to_floats(lin8)  # transmission_time, fit_interval

        col = [0.0]*30
        # --- Строка 1 ---
        col[0]  = float(prn)
        col[1]  = af0
        col[2]  = af1
        col[3]  = af2

        # --- Строка 2 ---
        col[4]  = (L2[0] if len(L2)>0 else 0.0)  # IODE
        col[5]  = (L2[1] if len(L2)>1 else 0.0)  # Crs
        col[6]  = (L2[2] if len(L2)>2 else 0.0)  # Δn
        col[7]  = (L2[3] if len(L2)>3 else 0.0)  # M0

        # --- Строка 3 ---
        col[8]  = (L3[0] if len(L3)>0 else 0.0)  # Cuc
        col[9]  = (L3[1] if len(L3)>1 else 0.0)  # e
        col[10] = (L3[2] if len(L3)>2 else 0.0)  # Cus
        col[11] = (L3[3] if len(L3)>3 else 0.0)  # sqrtA

        # --- Строка 4 ---
        col[12] = (L4[0] if len(L4)>0 else 0.0)  # Toe
        col[13] = (L4[1] if len(L4)>1 else 0.0)  # Cic
        col[14] = (L4[2] if len(L4)>2 else 0.0)  # Omega0
        col[15] = (L4[3] if len(L4)>3 else 0.0)  # Cis

        # --- Строка 5 ---
        col[16] = (L5[0] if len(L5)>0 else 0.0)  # i0
        col[17] = (L5[1] if len(L5)>1 else 0.0)  # Crc
        col[18] = (L5[2] if len(L5)>2 else 0.0)  # omega
        col[19] = (L5[3] if len(L5)>3 else 0.0)  # Omegadot

        # --- Строка 6 ---
        col[20] = (L6[0] if len(L6)>0 else 0.0)  # IDOT
        col[21] = (L6[1] if len(L6)>1 else 0.0)  # code_on_L2
        col[22] = (L6[2] if len(L6)>2 else 0.0)  # GPS_week
        col[23] = (L6[3] if len(L6)>3 else 0.0)  # L2flag

        # --- Строка 7 ---
        col[24] = (L7[0] if len(L7)>0 else 0.0)  # sv_accuracy
        col[25] = (L7[1] if len(L7)>1 else 0.0)  # sv_health
        col[26] = (L7[2] if len(L7)>2 else 0.0)  # TGD
        col[27] = (L7[3] if len(L7)>3 else 0.0)  # IODC

        # --- Строка 8 ---
        col[28] = (L8[0] if len(L8)>0 else 0.0)  # transmission_time (t_Tx)
        col[29] = (L8[1] if len(L8)>1 else 0.0)  # fit_interval (часов)

        cols.append(col)

    if not cols:
        return []
    n = len(cols)
    # теперь транспонируем 30×N
    eRdy = [[cols[j][i] for j in range(n)] for i in range(30)]
    return eRdy

def read_gps_ephemeris(file_nav):
    """
      1-я строка:  PRN, af0, af1, af2
      2-я строка:  IODE, Crs, Δn, M0
      3-я строка:  Cuc, e, Cus, sqrtA
      4-я строка:  Toe, Cic, Ω0, Cis
      5-я строка:  i0, Crc, ω, Ω̇
      6-я строка:  IDOT, code_on_L2, GPS_week, L2flag
      7-я строка:  sv_accuracy, sv_health, TGD, IODC
      8-я строка:  transmission_time, fit_interval
    """
    eRdy = rinex_get_nav_gps(file_nav)
    if not eRdy:
        return [], [], []

    # toe хранится в eRdy[12]
    toes = sorted({eRdy[12][c] for c in range(len(eRdy[0]))})
    # неделя GPS в eRdy[22]
    ephT = [[toe, int(eRdy[22][0]) if eRdy[22] else 0] for toe in toes]

    eph = [[None for _ in range(33)] for _ in range(len(toes))]
    ephSats = [[] for _ in range(len(toes))]

    for k, toe in enumerate(toes):
        cols = [n for n in range(len(eRdy[0])) if eRdy[12][n] == toe]
        nSat = len(cols)
        sats = []
        for n in cols:
            G = int(eRdy[0][n])   # PRN
            sats.append(G)
            r = GPSRec()
            r.nSat = nSat
            r.prn = G

            # - строка 1 -
            r.af0 = eRdy[1][n]
            r.af1 = eRdy[2][n]
            r.af2 = eRdy[3][n]

            # - строка 2 -
            r.iode   = eRdy[4][n]
            r.crs    = eRdy[5][n]
            r.deltan = eRdy[6][n]
            r.m0     = eRdy[7][n]

            # - строка 3 -
            r.cuc   = eRdy[8][n]
            r.e     = eRdy[9][n]
            r.cus   = eRdy[10][n]
            r.sqrtA = eRdy[11][n]

            # - строка 4 -
            r.toe     = eRdy[12][n]
            r.cic     = eRdy[13][n]
            r.omega0  = eRdy[14][n]
            r.cis     = eRdy[15][n]

            # - строка 5 -
            r.i0       = eRdy[16][n]
            r.crc      = eRdy[17][n]
            r.omega    = eRdy[18][n]
            r.omegaDot = eRdy[19][n]

            # - строка 6 -
            r.iDot      = eRdy[20][n]
            r.codeL2    = eRdy[21][n]
            r.gpsWeekNo = int(eRdy[22][n])
            r.L2flag    = eRdy[23][n]

            # - строка 7 -
            r.svaccur  = eRdy[24][n]
            r.svhealth = eRdy[25][n]
            r.tgd      = eRdy[26][n]
            r.iodc     = eRdy[27][n]

            # - строка 8 -
            r.txTime = eRdy[28][n]
            r.fitInt = eRdy[29][n]

            eph[k][G] = r
        ephSats[k] = sats
    return eph, ephSats, ephT

def rinex_get_nav_glonass(file_nav):
    """
    Читает возвращает матрицу 
      0: PRN        1: tauN       2: gammaN    3: tk
      4: x          5: dx         6: ddx       7: freqNo
      8: y          9: dy        10: ddy      11: age
     12: z         13: dz        14: ddz      15: svhealth
    """

    # Считываем всё тело файла (без заголовка)
    body = _read_body(file_nav)
    cols = []  # сюда добавляем столбцы (по одному блоку спутника)
    i = 0
    while i < len(body):
        # --- пропускаем пустые или нерелевантные строки ---
        if not body[i].strip():
            i += 1
            continue
        if body[i][0] != 'R':  # не ГЛОНАСС
            i += 1
            continue
        if i + 3 >= len(body):  # блок не полный
            break

        # --- читаем 4 строки блока спутника ---
        l1 = body[i]
        l2 = body[i+1]
        l3 = body[i+2]
        l4 = body[i+3]
        i += 4

        # --- конвертируем строки в массивы чисел ---
        prn = int(l1[1:3])         # номер спутника (Rxx → xx)
        L1 = _to_floats(l1[22:])   # строка 1: τN, γN, tk
        L2 = _to_floats(l2)        # строка 2: x, ẋ, ẍ, freqNo
        L3 = _to_floats(l3)        # строка 3: y, ẏ, ÿ, age
        L4 = _to_floats(l4)        # строка 4: z, ż, z̈, svhealth

        # --- создаём список значений для одного спутника (16 параметров) ---
        col = [0.0]*16

        # ===== Строка 1: часы спутника =====
        col[0] = float(prn)                    # R-номер
        col[1] = L1[0] if len(L1)>0 else 0.0   # τN — смещение шкалы времени (с)
        col[2] = L1[1] if len(L1)>1 else 0.0   # γN — относительная поправка частоты
        col[3] = L1[2] if len(L1)>2 else 0.0   # tk  — время момента измерения (с)

        # ===== Строка 2: координаты X =====
        col[4] = L2[0] if len(L2)>0 else 0.0   # X  (м)
        col[5] = L2[1] if len(L2)>1 else 0.0   # Ẋ (м/с)
        col[6] = L2[2] if len(L2)>2 else 0.0   # Ẍ (м/с²)
        col[7] = float(int(round(L2[3]))) if len(L2)>3 else 0.0  # freqNo (−7 … +6)

        # ===== Строка 3: координаты Y =====
        col[8]  = L3[0] if len(L3)>0 else 0.0  # Y  (м)
        col[9]  = L3[1] if len(L3)>1 else 0.0  # Ẏ (м/с)
        col[10] = L3[2] if len(L3)>2 else 0.0  # Ÿ (м/с²)
        col[11] = L3[3] if len(L3)>3 else 0.0  # age — возраст эфемерид (с)

        # ===== Строка 4: координаты Z =====
        col[12] = L4[0] if len(L4)>0 else 0.0  # Z  (м)
        col[13] = L4[1] if len(L4)>1 else 0.0  # Ż (м/с)
        col[14] = L4[2] if len(L4)>2 else 0.0  # Z̈ (м/с²)
        col[15] = L4[3] if len(L4)>3 else 0.0  # svhealth — флаг исправности (0 = ок)

        # добавляем запись в общий список
        cols.append(col)


    if not cols:
        return []

    # --- транспонируем (16×N) 
    n = len(cols)
    eRdyR = [[cols[j][i] for j in range(n)] for i in range(16)]
    return eRdyR

def read_glonass_ephemeris(file_nav):
    """
        0: PRN, 1: tauN, 2: gammaN, 3: tk,
        4: x,   5: dx,   6: ddx,    7: freqNo,
        8: y,   9: dy,  10: ddy,   11: age,
       12: z,  13: dz,  14: ddz,   15: svhealth
    """
    eRdyR = rinex_get_nav_glonass(file_nav)
    if not eRdyR:
        return [], [], []

    tks = sorted({eRdyR[3][c] for c in range(len(eRdyR[0]))})

    ephR     = [[None for _ in range(33)] for _ in range(len(tks))]  #
    ephRSats = [[] for _ in range(len(tks))]
    ephRT    = [[tk, 0] for tk in tks]  

    # Проходим по эпохам tk и собираем спутники этой эпохи
    for k, tk in enumerate(tks):
        # Индексы столбцов, относящихся к текущей эпохе tk
        cols = [n for n in range(len(eRdyR[0])) if eRdyR[3][n] == tk]
        nSat = len(cols)
        sats = []

        for n in cols:
            R = int(eRdyR[0][n])  # PRN 
            sats.append(R)

            r = GloRec()
            r.nSat = nSat
            r.prn  = R

            #  строка 1: tauN, gammaN, tk 
            r.tauN   = eRdyR[1][n]
            r.gammaN = eRdyR[2][n]
            r.tk     = eRdyR[3][n]

            #  строка 2: x, dx, ddx, freqNo 
            r.x      = eRdyR[4][n]
            r.dx     = eRdyR[5][n]
            r.ddx    = eRdyR[6][n]
            r.freqNo = int(eRdyR[7][n])

            #  строка 3: y, dy, ddy, age 
            r.y   = eRdyR[8][n]
            r.dy  = eRdyR[9][n]
            r.ddy = eRdyR[10][n]
            r.age = eRdyR[11][n]

            #  строка 4: z, dz, ddz, svhealth 
            r.z        = eRdyR[12][n]
            r.dz       = eRdyR[13][n]
            r.ddz      = eRdyR[14][n]
            r.svhealth = eRdyR[15][n]

            r.debugId = R
            ephR[k][R] = r

        ephRSats[k] = sats

    return ephR, ephRSats, ephRT


def load_nav_separate(GPS_FILE, GLO_FILE):
    """
    Загружает и парсит отдельные файлы эфемерид GPS и ГЛОНАСС.
    Возвращает кортеж из двух наборов:
        (gps, gpsSats, gpsT), (glo, gloSats, gloT)
    """
    gps, gpsSats, gpsT = read_gps_ephemeris(GPS_FILE)
    glo, gloSats, gloT = read_glonass_ephemeris(GLO_FILE)
    return (gps, gpsSats, gpsT), (glo, gloSats, gloT)



if __name__ == "__main__":
    # --- Файлы эфемерид ---
    GPS_FILE = "BRDC0010.17n"
    GLO_FILE = "base_01.rnx"  

    # Загружаем данные
    (gps, gpsSats, gpsT), (glo, gloSats, gloT) = load_nav_separate(GPS_FILE, GLO_FILE)

        
    # Вывод всех доступных НКА (спутников)

    gps_all = sorted({j for sats in gpsSats for j in sats})
    glo_all = sorted({j for sats in gloSats for j in sats}) if gloSats else []

    print("\n=== СПУТНИКИ GPS  ===")
    print(gps_all if gps_all else "(нет GPS данных)")

    print("\n=== СПУТНИКИ ГЛОНАСС  ===")
    print(glo_all if glo_all else "(нет ГЛОНАСС данных)")

