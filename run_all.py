from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
import re
import importlib
from GPS import parse_gps_block
from GLONASS import parse_glo_block

# импортируем 1.py, он отвечает за интерактивный выбор
nav_selector = importlib.import_module("1")

# регулярка для подсчёта чисел в блоках
_FLOAT_RE = re.compile(r'[+\-]?(?:\d+\.\d*|\.\d+)(?:[EeDd][+\-]?\d+)?')

def _detect_system_from_block(block_lines: list[str]) -> str:
    """Определяем систему (GPS или ГЛОНАСС) по структуре блока."""
    n_lines = len(block_lines)
    if n_lines >= 7:
        return "GPS"
    if n_lines <= 5:
        return "GLONASS"
    count = 0
    for ln in block_lines:
        s = ln.replace("D", "E").replace("d", "e")
        count += len(list(_FLOAT_RE.finditer(s)))
    return "GPS" if count >= 25 else "GLONASS"

def choose_and_parse() -> Optional[Tuple[str, str, Dict[str, Any], str]]:
    """
    Вызывает интерактив из 1.py (меню и ввод),
    возвращает:
      (sv_code, epoch_key, vals_dict, system)
    """
    pick = nav_selector.select_block_interactive()
    if pick is None:
        return None

    sv_code, epoch_key, block_lines, _system_name_from_1 = pick
    system = _detect_system_from_block(block_lines)

    if system == "GPS":
        vals = parse_gps_block(block_lines)
    else:
        vals = parse_glo_block(block_lines)

    return sv_code, epoch_key, vals, system


if __name__ == "__main__":
    pick = choose_and_parse()
    if pick is not None:
        sv_code, epoch_key, vals, system = pick

        if system == "GPS":
            af0 = vals["af0"]
            print("af0 (смещение часов) =", af0)
            af1 = vals["af1"]
            print("af1 (дрейф часов) =", af1)
            af2 = vals["af2"]
            print("af2 (ускорение дрейфа) =", af2)
            IODE = vals["IODE"]
            print("IODE (номер эфемерид) =", IODE)
            Crs = vals["Crs"]
            print("Crs (синусная поправка радиуса) =", Crs)
            dn = vals["dn"]
            print("dn (приращение средней угловой скорости) =", dn)
            M0 = vals["M0"]
            print("M0 (средняя аномалия) =", M0)
            Cuc = vals["Cuc"]
            print("Cuc (косинусная поправка аргумента широты) =", Cuc)
            e = vals["e"]
            print("e (эксцентриситет) =", e)
            Cus = vals["Cus"]
            print("Cus (синусная поправка аргумента широты) =", Cus)
            sqrtA = vals["sqrtA"]
            print("sqrtA (корень из большой полуоси) =", sqrtA)
            Toe = vals["Toe"]
            print("Toe (время эфемерид) =", Toe)
            Cic = vals["Cic"]
            print("Cic (косинусная поправка наклонения) =", Cic)
            OMEGA0 = vals["OMEGA0"]
            print("OMEGA0 (долгота восходящего узла) =", OMEGA0)
            Cis = vals["Cis"]
            print("Cis (синусная поправка наклонения) =", Cis)
            i0 = vals["i0"]
            print("i0 (наклонение на эпоху) =", i0)
            Crc = vals["Crc"]
            print("Crc (косинусная поправка радиуса) =", Crc)
            omega = vals["omega"]
            print("omega (аргумент перигея) =", omega)
            OMEGADOT = vals["OMEGADOT"]
            print("OMEGADOT (скорость изменения долготы узла) =", OMEGADOT)
            IDOT = vals["IDOT"]
            print("IDOT (скорость изменения наклонения) =", IDOT)
            L2_codes = vals["L2_codes"]
            print("L2_codes (L2 код) =", L2_codes)
            week = vals["week"]
            print("week (номер недели) =", week)
            L2P_flag = vals["L2P_flag"]
            print("L2P_flag (флаг L2P) =", L2P_flag)
            SVacc = vals["SVacc"]
            print("SVacc (точность эфемерид) =", SVacc)
            SVhealth = vals["SVhealth"]
            print("SVhealth (здоровье НКА) =", SVhealth)
            TGD = vals["TGD"]
            print("TGD (групповая задержка) =", TGD)
            IODC = vals["IODC"]
            print("IODC (Issue of Data Clock) =", IODC)
            t_trans = vals["t_trans"]
            print("t_trans (время передачи) =", t_trans)
            fit_int = vals["fit_int"]
            print("fit_int (интервал действия эфемерид) =", fit_int)



        elif system == "GLONASS":
            tauN = vals["tauN"]
            print("tauN (смещение часов) =", tauN)
            gammaN = vals["gammaN"]
            print("gammaN (дрейф частоты) =", gammaN)
            tk = vals["tk"]
            print("tk (время измерений) =", tk)
            x = vals["x"]
            print("x (координата X) =", x)
            vx = vals["vx"]
            print("vx (скорость X) =", vx)
            ax = vals["ax"]
            print("ax (ускорение X) =", ax)
            health = vals["health"]
            print("health (здоровье НКА) =", health)
            y = vals["y"]
            print("y (координата Y) =", y)
            vy = vals["vy"]
            print("vy (скорость Y) =", vy)
            ay = vals["ay"]
            print("ay (ускорение Y) =", ay)
            freq_num = vals["freq_num"]
            print("freq_num (частотный номер) =", freq_num)
            z = vals["z"]
            print("z (координата Z) =", z)
            vz = vals["vz"]
            print("vz (скорость Z) =", vz)
            az = vals["az"]
            print("az (ускорение Z) =", az)
            age = vals["age"]
            print("age (возраст эфемерид) =", age)
