from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
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
        # - Строка 1: PRN / EPOCH / SV CLK
        self.af0 = 0.0         # Сдвиг часов спутника (секунда)
        self.af1 = 0.0         # Скорость ухода часов (сек./сек.)
        self.af2 = 0.0         # Ускорение ухода часов (сек./сек.^2)
        # - Строка 2: BROADCAST ORBIT - 1 -
        self.iode   = 0.0      # IODE
        self.crs    = 0.0      # Crs (метр)
        self.deltan = 0.0      # Delta n (радиан/секунда)
        self.m0     = 0.0      # M0 (радиан)
        # - Строка 3: BROADCAST ORBIT - 2 -
        self.cuc   = 0.0       # Cuc (радиан)
        self.e     = 0.0       # Эксцентриситет орбиты (e)
        self.cus   = 0.0       # Cus (радиан)
        self.sqrtA = 0.0       # sqrt(A) (метр^0.5)
        # - Строка 4: BROADCAST ORBIT - 3 -
        self.toe     = 0.0     # Время эфемерид (Toe) (секунда от начала GPS-недели)
        self.cic     = 0.0     # Cic (радиан)
        self.OMEGA  = 0.0      # OMEGA (радиан)
        self.cis     = 0.0     # CIS (радиан)
        # - Строка 5: BROADCAST ORBIT - 4 -
        self.i0       = 0.0    # наклон орбиты спутника относительно плоскости экватора (радиан)
        self.crc      = 0.0    # Crc (метр)
        self.omega    = 0.0    # omega (радиан)
        self.omegaDot = 0.0    # OMEGA DOT (радиан/секунда)
        # - Строка 6: BROADCAST ORBIT - 5 -
        self.iDot     = 0.0    # IDOT (радиан/секунда)
        self.codeL2   = 0.0    # Коды в диапазоне L2
        self.gpsWeekNo= 0      # Номер GPS-недели
        self.L2flag   = 0.0    # Флаг данных L2 P
        # - Строка 7: BROADCAST ORBIT - 6 -
        self.svaccur  = 0.0    # Точность положения спутника (метр)
        self.svhealth = 0.0    # Исправность
        self.tgd      = 0.0    # TGD (секунда)
        self.iodc     = 0.0    # IODC
        # - Строка 8: BROADCAST ORBIT - 7 -
        self.txTime = 0.0      # Время передачи сообщения (секунды GPS недели)
        self.fitInt = 0.0      # Интервал аппроксимации орбиты (часы)
        # - доп. поле из твоего кода -
        self.toc = 0.0         # время часов (сек от начала суток), используется в обработке
        self.freqNo = 0        # для совместимости (не используется для GPS)
        self.debugId = 0       # ID для отладки (копия PRN)

class GloRec:
    def __init__(self):
        # - служебное -
        self.nSat = 0        # количество спутников в эпохе
        self.prn  = 0        # PRN
        # - строка 1: RPRN / EPOCH / SV CLK -
        self.tauN   = 0.0    # Сдвиг часов спутника (cекунды) (-TauN)
        self.gammaN = 0.0    # Относительный сдвиг частоты (+GammaN)
        self.tk     = 0.0    # Время сообщения (tk)
        # - строка 2: BROADCAST ORBIT 1 -
        self.x   = 0.0       # X координата спутника (км)
        self.dx  = 0.0       # скорость по X (км/с)
        self.ddx = 0.0       # ускорение по X (км/с2)
        self.svhealth = 0.0  # Исправность (0=OK)      
        # - строка 3: BROADCAST ORBIT 2 -
        self.y   = 0.0       # Y координата спутника (км)
        self.dy  = 0.0       # скорость по Y (км/с)
        self.ddy = 0.0       # ускорение по Y (км/с2)
        self.freqNo = 0      # Номер частотного канала (-7 ... +13)
        # - строка 4: BROADCAST ORBIT 3 -
        self.z   = 0.0       # Z координата спутника (км)
        self.dz  = 0.0       # скорость по Z (км/с)
        self.ddz = 0.0       # ускорение по Z (км/с2)
        self.age = 0.0       # Возраст информации (дней)
        # - дополнительное -
        self.debugId = 0     # ID для отладки (копия PRN)

def rinex_get_nav_gps(file_nav):
    """
    читаем и формирует столбцы в порядке, как в файле.
      0: PRN               1: af0             2: af1            3: af2
      4: IODE              5: Crs             6: Delta n        7: M0
      8: Cuc               9: e               10: Cus           11: sqrtA
      12: Toe              13: Cic            14: OMEGA        15: Cis
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

        # --- Строка 1 ---
        prn = int(lin1[0:3])               
        L1 = _to_floats(lin1[22:])         # преобразуем числа начиная с позиции 22 
        L2 = _to_floats(lin2)  
        L3 = _to_floats(lin3)  
        L4 = _to_floats(lin4)  
        L5 = _to_floats(lin5)  
        L6 = _to_floats(lin6)  
        L7 = _to_floats(lin7)  
        L8 = _to_floats(lin8)  

        col = [0.0]*30

        # --- Строка 1 ---
        col[0] = float(prn)
        col[1] = L1[0] if len(L1) > 0 else 0.0  
        col[2] = L1[1] if len(L1) > 1 else 0.0  
        col[3] = L1[2] if len(L1) > 2 else 0.0  

        # --- Строка 2 ---
        col[4]  = (L2[0] if len(L2)>0 else 0.0)  
        col[5]  = (L2[1] if len(L2)>1 else 0.0) 
        col[6]  = (L2[2] if len(L2)>2 else 0.0)  
        col[7]  = (L2[3] if len(L2)>3 else 0.0)  

        # --- Строка 3 ---
        col[8]  = (L3[0] if len(L3)>0 else 0.0)  
        col[9]  = (L3[1] if len(L3)>1 else 0.0)  
        col[10] = (L3[2] if len(L3)>2 else 0.0)  
        col[11] = (L3[3] if len(L3)>3 else 0.0)  

        # --- Строка 4 ---
        col[12] = (L4[0] if len(L4)>0 else 0.0)  
        col[13] = (L4[1] if len(L4)>1 else 0.0)  
        col[14] = (L4[2] if len(L4)>2 else 0.0)  
        col[15] = (L4[3] if len(L4)>3 else 0.0)  

        # --- Строка 5 ---
        col[16] = (L5[0] if len(L5)>0 else 0.0)  
        col[17] = (L5[1] if len(L5)>1 else 0.0)  
        col[18] = (L5[2] if len(L5)>2 else 0.0)  
        col[19] = (L5[3] if len(L5)>3 else 0.0)  

        # --- Строка 6 ---
        col[20] = (L6[0] if len(L6)>0 else 0.0)  
        col[21] = (L6[1] if len(L6)>1 else 0.0)  
        col[22] = (L6[2] if len(L6)>2 else 0.0)  
        col[23] = (L6[3] if len(L6)>3 else 0.0)  

        # --- Строка 7 ---
        col[24] = (L7[0] if len(L7)>0 else 0.0)  
        col[25] = (L7[1] if len(L7)>1 else 0.0)  
        col[26] = (L7[2] if len(L7)>2 else 0.0)  
        col[27] = (L7[3] if len(L7)>3 else 0.0)  

        # --- Строка 8 ---
        col[28] = (L8[0] if len(L8)>0 else 0.0)  
        col[29] = (L8[1] if len(L8)>1 else 0.0)

        cols.append(col)

    if not cols:
        return []
    n = len(cols)
    # теперь транспонируем 30×N
    eRdy = [[cols[j][i] for j in range(n)] for i in range(30)]
    return eRdy

def read_gps_ephemeris(file_nav):
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
            r.OMEGA  = eRdy[14][n]
            r.cis     = eRdy[15][n]

            # - строка 5 -
            r.i0       = eRdy[16][n]
            r.crc      = eRdy[17][n]
            r.omega    = eRdy[18][n]
            r.omegaDot = eRdy[19][n]

            # - строка 6 -
            r.iDot      = eRdy[20][n]
            r.codeL2    = eRdy[21][n]
            r.gpsWeekNo = eRdy[22][n]
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
    читаем и формирует столбцы в порядке, как в файле.
     0: PRN, 1: tauN, 2: gammaN, 3: tk,
     4: x,   5: dx,   6: ddx,    7: svhealth,
     8: y,   9: dy,  10: ddy,   11: freqNo,
     12: z,  13: dz,  14: ddz,   15:age 
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

        if i + 3 >= len(body):  # блок не полный
            break

        # --- читаем 4 строки блока спутника ---
        l1 = body[i]
        l2 = body[i+1]
        l3 = body[i+2]
        l4 = body[i+3]
        i += 4

        # --- конвертируем строки в массивы чисел ---
        prn = int(l1[0:3])         
        L1 = _to_floats(l1[22:])   
        L2 = _to_floats(l2)        
        L3 = _to_floats(l3)        
        L4 = _to_floats(l4)        

        col = [0.0]*16

        # ===== Строка 1:  =====
        col[0] = float(prn)                    
        col[1] = L1[0] if len(L1)>0 else 0.0   
        col[2] = L1[1] if len(L1)>1 else 0.0   
        col[3] = L1[2] if len(L1)>2 else 0.0   

        # ===== Строка 2: координаты X =====
        col[4] = L2[0] if len(L2)>0 else 0.0   
        col[5] = L2[1] if len(L2)>1 else 0.0   
        col[6] = L2[2] if len(L2)>2 else 0.0   
        col[11] = L3[3] if len(L3) > 3 else 0.0  

        # ===== Строка 3: координаты Y =====
        col[8]  = L3[0] if len(L3)>0 else 0.0  
        col[9]  = L3[1] if len(L3)>1 else 0.0  
        col[10] = L3[2] if len(L3)>2 else 0.0  
        col[11] = L3[3] if len(L3)>3 else 0.0  

        # ===== Строка 4: координаты Z =====
        col[12] = L4[0] if len(L4)>0 else 0.0  
        col[13] = L4[1] if len(L4)>1 else 0.0 
        col[14] = L4[2] if len(L4)>2 else 0.0  
        col[15] = L4[3] if len(L4)>3 else 0.0  
        
        # добавляем запись в общий список
        cols.append(col)

    if not cols:
        return []

    # --- транспонируем (16×N) 
    n = len(cols)
    eRdyR = [[cols[j][i] for j in range(n)] for i in range(16)]
    return eRdyR

def read_glonass_ephemeris(file_nav):
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

            #  строка 1: 
            r.tauN     = eRdyR[1][n]
            r.gammaN   = eRdyR[2][n]
            r.tk       = eRdyR[3][n]

            #  строка 2: 
            r.x        = eRdyR[4][n]
            r.dx       = eRdyR[5][n]
            r.ddx      = eRdyR[6][n]
            r.svhealth = eRdyR[7][n]

            #  строка 3:  
            r.y        = eRdyR[8][n]
            r.dy       = eRdyR[9][n]
            r.ddy      = eRdyR[10][n]
            r.freqNo   = eRdyR[11][n]

            #  строка 4: 
            r.z        = eRdyR[12][n]
            r.dz       = eRdyR[13][n]
            r.ddz      = eRdyR[14][n]
            r.age      = eRdyR[15][n]


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

#==========================================================================================================================================
# Константы для GPS 
MU = 3.986005e14         # м^3/с^2
# Угловая скорость вращения Земли 
OMEGA_E = 7.292115e-5     # рад/с

def wrap_week(tk):
    if tk > 302400:   # если прошло больше половины недели вперёд
        tk -= 604800  # считаем, что это "−X секунд" от предыдущей недели
    if tk < -302400:  # если ушли более чем на полнедели назад
        tk += 604800  # добавляем неделю вперёд
    return tk

def solve_kepler(M, e, iters: int = 10) -> float:
    """
    Решает уравнение Кеплера M = E - e*sin(E) итерационным методом Ньютона.
    Используется для вычисления эксцентрической аномалии E по известной
    средней аномалии M и эксцентриситету e.
    Пошагово:
    1. Задаём начальное приближение E0 = M (достаточно при e < 0.1)
    2. Повторяем N итераций:
           E_{n+1} = E_n + (M - (E_n - e*sin(E_n))) / (1 - e*cos(E_n))
    3. Возвращаем итоговое значение E (в радианах)
    """
    E = M  # начальное приближение
    for _ in range(iters):
        delta = (M - (E - e * np.sin(E))) / (1 - e * np.cos(E))
        E += delta
        # Можно прервать итерации, если изменение уже мало:
        if abs(delta) < 1e-12:
            break
    return E

def ecef_from_eph(eph, t):
    """
    Вычисляет ECEF-координаты GPS-спутника по переданным эфемеридам LNAV в 
    момент времени t (сек GPS-недели).
    Параметры eph:
      eph.sqrtA  : sqrt(A), корень из большой полуоси [м^0.5]
      eph.e      : эксцентриситет
      eph.deltan : поправка к средней угл. скорости [рад/с]
      eph.m0     : средняя аномалия в toe [рад]
      eph.omega  : аргумент перигея ω [рад]
      eph.cus,cuc,crs,crc,cis,cic : гармонические поправки
      eph.i0     : наклон i0 [рад], eph.iDot: di/dt [рад/с]
      eph.OMEGA  : долгота восходящего узла Ω0 в toe [рад]
      eph.omegaDot : dΩ/dt [рад/с]
      eph.toe    : эпоха эфемерид toe [с]

    Возвращает:
      numpy.array([x, y, z]) в ECEF, метры.
    """
    # === 1) Базовые величины ===
    # Большая полуось A = (sqrtA)^2
    A = eph.sqrtA ** 2

    # Номинальная средняя угловая скорость (n0 = sqrt(mu / A^3))
    n0 = np.sqrt(MU / A**3)

    # Разность времен tk = t - toe, с учетом перехода через границу недели
    tk = wrap_week(t - eph.toe)

    # Исправленная средняя угловая скорость n = n0 + Δn
    n = n0 + eph.deltan

    # Средняя аномалия в момент t: Mk = M0 + n*tk
    Mk = eph.m0 + n * tk

    # === 2) Уравнение Кеплера и истинная аномалия ===
    # Решаем M = E - e*sin(E) → получаем эксцентрическую аномалию Ek
    Ek = solve_kepler(Mk, eph.e)

    # Истинная аномалия ν из Ek (устойчивая формула через tan(E/2))
    vk = 2 * np.arctan2(
        np.sqrt(1 + eph.e) * np.sin(Ek / 2.0),
        np.sqrt(1 - eph.e) * np.cos(Ek / 2.0)
    )

    # === 3) Гармоники второй гармоники и исправления радиуса/широты/наклона ===
    # Аргумент широты без поправок: Φk = νk + ω
    phi_k = vk + eph.omega

    # Поправка аргумента широты u = Φ + δu
    du = eph.cus * np.sin(2 * phi_k) + eph.cuc * np.cos(2 * phi_k)

    # Поправка радиуса r = A(1 - e*cosE) + δr
    dr = eph.crs * np.sin(2 * phi_k) + eph.crc * np.cos(2 * phi_k)

    # Поправка наклона i = i0 + iDot*tk + δi
    di = eph.cis * np.sin(2 * phi_k) + eph.cic * np.cos(2 * phi_k)

    u = phi_k + du
    r = A * (1 - eph.e * np.cos(Ek)) + dr
    i = eph.i0 + eph.iDot * tk + di

    # === 4) Переход из плоскости орбиты к ECEF ===
    # Координаты в орбитальной плоскости
    x_orb = r * np.cos(u)
    y_orb = r * np.sin(u)

    # Исправленная долгота восходящего узла:
    # Ωk = Ω0 + (Ωdot - Ωe)*tk - Ωe*toe
    #  (учёт суточного вращения Земли Ωe и дрейфа узла)
    Omega_k = eph.OMEGA + eph.omegaDot * tk  # без учёта вращения Земли

    # Преобразование в ECEF (WGS-84)
    x = x_orb * np.cos(Omega_k) - y_orb * np.cos(i) * np.sin(Omega_k)
    y = x_orb * np.sin(Omega_k) + y_orb * np.cos(i) * np.cos(Omega_k)
    z = y_orb * np.sin(i)

    return np.array([x, y, z])

def track_for_prn(eph, hours = 12.0, step_track_s_GPS= 300.0):
    """
    Строит траекторию спутника (PRN) в системе ECEF по эфемеридам GPS.
    Параметры:
      eph   : объект с эфемеридами 
      hours : длительность интервала (в часах) вокруг toe
      step_track_s_GPS: шаг дискретизации по времени (в секундах)
    Возвращает:
      ts  : массив времён (секунды GPS-недели)
      xyz : массив координат [x, y, z] в метрах (ECEF)
    """
    # Начало интервала = toe - половина диапазона 
    t0 = eph.toe - (hours * 3600) / 2
    # Массив моментов времени с равным шагом step_track_s_GPS
    ts = np.arange(t0, t0 + hours * 3600 + step_track_s_GPS, step_track_s_GPS)
    # Для каждого момента t считаем координаты спутника в системе ECEF
    xyz = np.array([ecef_from_eph(eph, t) for t in ts])
    # Возвращаем массив времён и соответствующих координат
    return ts, xyz

#===========================================================================================================================================
#Константы для ГЛОНАСС
MU_PZ = 398600.4418e9   # гравитационный параметр Земли, м^3/с^2
AE_PZ = 6378136.0       # экваториальный радиус Земли, м
J2_PZ = 1082625.75e-9   # коэффициент сплюснутости (вторая гармоника), безразмерный

def _grav_j2_acc(x, y, z):
    """
    Вычисляет ускорение (ax, ay, az) спутника в геоцентрической системе
     с учётом второй гармоники J2 (сплюснутости Земли).
    """
    # Расстояние до центра Земли и вспомогательные параметры
    r2 = x*x + y*y + z*z      # квадрат расстояния
    r  = np.sqrt(r2)          # модуль радиус-вектора
    zx = z*z / r2             # отношение (z/r)^2
    # Коэффициент с поправкой на J2
    k = 1.5 * J2_PZ * (AE_PZ**2) / (r2)
    # Основной множитель
    mu_r3 = MU_PZ / (r2 * r)
    # Поправки для каждой координаты
    c = 1 - k * (5*zx - 1)
    ax = -mu_r3 * x * c
    ay = -mu_r3 * y * c
    az = -mu_r3 * z * (1 - k * (5*zx - 3))

    return ax, ay, az

def _rotating_frame_acc(x, y, z, vx, vy, vz):
    """
    Вариант БЕЗ учёта вращения Земли.
    Вращающаяся система → считаем инерциальной, поэтому
    дополнительные ускорения (центробежное и кориолисово) = 0.
    """
    return 0.0, 0.0, 0.0

def _glo_rhs(state, add_ax=0.0, add_ay=0.0, add_az=0.0):
    """
    state = [x, y, z, vx, vy, vz]
        x, y, z  — координаты спутника (м)
        vx, vy, vz — скорости (м/с)
    add_ax, add_ay, add_az — дополнительные ускорения
        (лунно–солнечные возмущения, постоянные на ±15 мин)
    Возвращает вектор:
        [dx/dt, dy/dt, dz/dt, d²x/dt², d²y/dt², d²z/dt²]
    """
    # Распаковываем состояние
    x, y, z, vx, vy, vz = state

    # --- 1) Гравитация с учётом сплюснутости Земли (J2) ---
    ax_g, ay_g, az_g = _grav_j2_acc(x, y, z)

    # --- 2) Центробежное + кориолисово ускорения во вращающейся системе ---
    ax_r, ay_r, az_r = _rotating_frame_acc(x, y, z, vx, vy, vz)

    # --- 3) Добавляем возмущения от Луны и Солнца ---
    ax = ax_g + ax_r + add_ax
    ay = ay_g + ay_r + add_ay
    az = az_g + az_r + add_az

    return np.array([vx, vy, vz, ax, ay, az])

def _rk4_step(state, dt, ax_add, ay_add, az_add):
    """
    Параметры:
      state  – вектор состояния [x, y, z, vx, vy, vz]
      dt     – шаг интегрирования (сек)
      ax_add, ay_add, az_add – дополнительные ускорения (лунно–солнечные)
    Возвращает:
      новое состояние после шага dt
    """
    # k1: производные при начальном состоянии
    k1 = _glo_rhs(state, ax_add, ay_add, az_add)
    # k2: производные в середине шага (с учётом k1)
    k2 = _glo_rhs(state + 0.5 * dt * k1, ax_add, ay_add, az_add)
    # k3: ещё одно приближение середины шага (с учётом k2)
    k3 = _glo_rhs(state + 0.5 * dt * k2, ax_add, ay_add, az_add)
    # k4: производные в конце шага (с учётом k3)
    k4 = _glo_rhs(state + dt * k3, ax_add, ay_add, az_add)
    # Итоговое новое состояние (взвешенное среднее 1–2–2–1)
    state_next = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return state_next

def glo_ecef_from_icd(ephR, t, step_s=10.0):
    """
    Вычисляет координаты спутника ГЛОНАСС (ECEF)
    Параметры:
        ephR   — объект с эфемеридами спутника (x, y, z, dx, dy, dz, ddx, ddy, ddz, tk)
        t      — момент времени, на который требуется определить координаты, [с]
        step_s — шаг интегрирования, [с] 
    Возвращает:
        numpy.array([x, y, z]) — координаты спутника в метрах 
    """
    # --- 1) Начальные условия из эфемерид ---
    x0  = ephR.x  * 1000.0    # координата X в метрах
    y0  = ephR.y  * 1000.0    # координата Y в метрах
    z0  = ephR.z  * 1000.0    # координата Z в метрах
    vx0 = ephR.dx * 1000.0    # скорость по X в м/с
    vy0 = ephR.dy * 1000.0    # скорость по Y в м/с
    vz0 = ephR.dz * 1000.0    # скорость по Z в м/с
    # --- 2) Лунно–солнечные ускорения ---
    # В ИКД указано, что ускорения ddx, ddy, ddz считаются постоянными
    ax_add = ephR.ddx * 1000.0
    ay_add = ephR.ddy * 1000.0
    az_add = ephR.ddz * 1000.0
    # --- 3) Расчёт временного интервала от эпохи tk ---
    tk = ephR.tk                # эпоха, для которой заданы параметры эфемерид
    dt_total = float(t - tk)    # общее смещение по времени, [с]

    # Если момент совпадает с tk — возвращаем исходные координаты без интегрирования
    if dt_total == 0.0:
        return np.array([x0, y0, z0])

    # --- 4) Разбиваем интервал на шаги интегрирования ---
    # Определяем количество шагов интегратора, чтобы равномерно пройти весь интервал
    nsteps = int(np.ceil(abs(dt_total) / step_s))  # количество шагов
    dt = dt_total / nsteps                         # реальный шаг по времени
    # --- 5) Формируем начальный вектор состояния ---
    state = np.array([x0, y0, z0, vx0, vy0, vz0], dtype=float)
    # --- 6) Интегрирование уравнений движения ---
    for _ in range(nsteps):
        state = _rk4_step(state, dt, ax_add, ay_add, az_add)
    # --- 7) Возвращаем координаты после интегрирования ---
    # Берём только первые три компоненты (x, y, z),
    # скорости и ускорения для построения орбиты не нужны.
    return state[:3]

def track_for_prn_glo(ephR, hours=2.0, step_track_s_glo=60.0):
    """
    Строит траекторию спутника ГЛОНАСС 
    Параметры:
        ephR          — структура с эфемеридами спутника 
        hours         — длительность интервала в часах 
        step_track_s  — шаг между точками траектории, секунд 

    Возвращает:
        ts  — массив временных отметок [с]
        xyz — массив координат [x, y, z] в метрах (ПЗ-90.02, ECEF)
    """
    # ---  Определяем начало интервала по времени ---
    t0 = ephR.tk - (hours * 3600) / 2

    # --- Формируем массив времён для расчёта координат ---
    ts = np.arange(t0, t0 + hours * 3600 + step_track_s_glo, step_track_s_glo)

    # --- Для каждого момента времени вызываем численный расчёт положения спутника ---
    xyz = np.array([glo_ecef_from_icd(ephR, t) for t in ts])

    return ts, xyz

def plot_orbits_all_systems(gps, gps_idx, glo, glo_idx, prn_list):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Земля
    R_E = 6378137.0
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = R_E*np.outer(np.cos(u), np.sin(v))
    ys = R_E*np.outer(np.sin(u), np.sin(v))
    zs = R_E*np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, linewidth=0, alpha=0.25)

    # Цвет — по PRN
    cmap = mpl.colormaps.get_cmap('tab20')
    prn_max = max(prn_list) if prn_list else 1
    seen_labels = set()

    # GPS (сплошные)
    for k in gps_idx:
        for prn in prn_list:
            sat = gps[k][prn] if (0 <= k < len(gps)) else None
            if sat is None:
                continue
            _, trk = track_for_prn(sat) 
            color = cmap((prn - 1) / prn_max)
            label = f"G{prn:02d} (эпоха {k})"
            h, = ax.plot(trk[:, 0], trk[:, 1], trk[:, 2], color=color, linestyle='-')
            if label not in seen_labels:
                h.set_label(label); seen_labels.add(label)

    # ГЛОНАСС (пунктир)
    for k in glo_idx:
        for prn in prn_list:
            sat = glo[k][prn] if (0 <= k < len(glo)) else None
            if sat is None:
                continue
            _, trk = track_for_prn_glo(sat)  
            color = cmap((prn - 1) / prn_max)
            label = f"R{prn:02d} (эпоха {k})"
            h, = ax.plot(trk[:, 0], trk[:, 1], trk[:, 2], color=color, linestyle='--')
            if label not in seen_labels:
                h.set_label(label); seen_labels.add(label)

    ax.set_xlabel("X, м"); ax.set_ylabel("Y, м"); ax.set_zlabel("Z, м")
    ax.set_title("Орбиты GPS (сплошные) и ГЛОНАСС (пунктир)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout(); plt.show()

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

    print("\n=== ЭПОХИ GPS ===")
    for k, (toe, week) in enumerate(gpsT):
        print(f"{k:2d}: Toe = {toe:.1f} s, Week = {week}")

    print("\n=== ЭПОХИ ГЛОНАСС ===")
    for k, (tk, _) in enumerate(gloT):
        print(f"{k:2d}: tk = {tk:.1f} s")

    # === ПРОВЕРКА ОТДЕЛЬНЫХ ЭПОХ И СПУТНИКОВ ===
    # задай интересующие индексы
    gps_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,1,16,17,18,19,20,21,22,23,24]   # индексы эпох GPS 
    glo_idx = [0,1,2,3,4 ]   # индексы эпох ГЛОНАСС
    prn_list = [5,6]  # номера спутников для вывода

    print("\n=== ВЫБРАННЫЕ ЭПОХИ GPS ===")
    for k in gps_idx:
        if k >= len(gps):
            continue
        toe, week = gpsT[k]
        print(f"--- Epoch #{k}: Toe={toe:.1f} s, Week={week}")
        for prn in prn_list:
            sat = gps[k][prn]
            if sat is None:
                continue
            print(f"G{prn:02d}: e={sat.e:.6f}, sqrtA={sat.sqrtA:.3f}, "
                f"omega={sat.omega:.3f}, i0={sat.i0:.3f}, Toe={sat.toe:.1f}")

    print("\n=== ВЫБРАННЫЕ ЭПОХИ ГЛОНАСС ===")
    for k in glo_idx:
        if k >= len(glo):
            continue
        tk, _ = gloT[k]
        print(f"--- Epoch #{k}: tk={tk:.1f} s")
        for prn in prn_list:
            sat = glo[k][prn]
            if sat is None:
                continue
            print(f"R{prn:02d}: x={sat.x:.1f}, y={sat.y:.1f}, z={sat.z:.1f}, "
                f"dx={sat.dx:.4f}, dy={sat.dy:.4f}, dz={sat.dz:.4f}, freqNo={sat.freqNo}")
        import numpy as np

    # === Совмещённый график для GPS + ГЛОНАСС ===
    plot_orbits_all_systems(gps, gps_idx, glo, glo_idx, prn_list)