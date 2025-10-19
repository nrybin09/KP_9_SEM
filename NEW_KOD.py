from pathlib import Path
from datetime import datetime
from math import sin, cos, sqrt
import re

_FLOAT_RE = re.compile(r'[+\-]?(?:\d+\.\d*|\.\d+)(?:[EeDd][+\-]?\d+)?') # для поиска числа с плавующей точкой
FILENAMES = ["base_01.rnx", "BRDC0010.17n"]

def read_body(fp):#пропускаем заголовок файла 
    lines = Path(fp).read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, ln in enumerate(lines[:800]):
        if "END OF HEADER" in ln:
            return lines[i+1:]
    return lines

def is_block_start(tokens):#пустая строка 
    if not tokens: 
        return False 
    t0 = tokens[0]
    return t0.isdigit()

def to_floats(s):#преобразование в формат понятного для питона
    s = s.replace('D', 'E').replace('d', 'E')
    return [float(m.group(0)) for m in _FLOAT_RE.finditer(s)]

def parse_epoch_prn_and_clock(line, default_sys="G"):#читаем первую сткоку для определения времени
    t = line.strip().split()
    sys = default_sys
    prn = int(t[0])
    y, m, d, H, M = map(int, t[1:6]); 
    s = float(t[6])
    clock = to_floats(" ".join(t[7:]))[:3]
    if y < 100: y = 2000 + y if y < 80 else 1900 + y
    return f"{sys}{prn:02d}", sys, datetime(y, m, d, H, M, int(round(s))), clock

def detect_block(lines, i):#определение типа G или R 
    n = len(lines)
    block = []
    for j in range(i, n):
        tok = lines[j].strip().split()
        if j > i and tok and tok[0].isdigit():
            break
        block.append(lines[j])
    n_lines = len(block)
    if n_lines >= 7:       # GPS 8 строк
        return 8
    if n_lines <= 5:       # ГЛОНАСС 4 строки
        return 4
    count = 0
    for ln in block:
        s = ln.replace("D", "E").replace("d", "E")  
        count += len(list(_FLOAT_RE.finditer(s)))

    return 8 if count >= 25 else 4

def parse_block(lines, i):
    prn, sys, epoch, clock = parse_epoch_prn_and_clock(lines[i])
    consumed = detect_block(lines, i)
    cols = [to_floats(lines[i+k]) for k in range(1, consumed )]
    flat = clock[:] + [v for row in cols for v in row]
    info = {"system": sys, "prn": prn, "epoch": epoch, "raw": flat}
    if consumed == 4 and sys != "R":
        try:
            sat_num = int(prn[1:] if prn[0].isalpha() else prn)
        except:
            sat_num = int(''.join(ch for ch in prn if ch.isdigit()))
        prn = f"R{sat_num:02d}"
        sys = "R"
        info["system"] = sys
        info["prn"] = prn

    if consumed >= 8:  
        l1 = cols[0] if len(cols) > 0 else []
        l2 = cols[1] if len(cols) > 1 else []
        l3 = cols[2] if len(cols) > 2 else []
        l4 = cols[3] if len(cols) > 3 else []
        l5 = cols[4] if len(cols) > 4 else []
        l6 = cols[5] if len(cols) > 5 else []
        l7 = cols[6] if len(cols) > 6 else []
        l8 = cols[7] if len(cols) > 7 else []
        gps = {
            "af0":  clock[0] if len(clock) > 0 else None,
            "af1":  clock[1] if len(clock) > 1 else None,
            "af2":  clock[2] if len(clock) > 2 else None,

            "IODE":   l1[0] if len(l1) > 0 else None,
            "Crs":    l1[1] if len(l1) > 1 else None,
            "dn":     l1[2] if len(l1) > 2 else None,
            "M0":     l1[3] if len(l1) > 3 else None,

            "Cuc":    l2[0] if len(l2) > 0 else None,
            "e":      l2[1] if len(l2) > 1 else None,
            "Cus":    l2[2] if len(l2) > 2 else None,
            "sqrtA":  l2[3] if len(l2) > 3 else None,

            "Toe":    l3[0] if len(l3) > 0 else None,
            "Cic":    l3[1] if len(l3) > 1 else None,
            "Omega0": l3[2] if len(l3) > 2 else None,
            "Cis":    l3[3] if len(l3) > 3 else None,

            "i0":       l4[0] if len(l4) > 0 else None,
            "Crc":      l4[1] if len(l4) > 1 else None,
            "omega":    l4[2] if len(l4) > 2 else None,
            "OmegaDot": l4[3] if len(l4) > 3 else None,

            "IDOT":    l5[0] if len(l5) > 0 else None,
            "codesL2": l5[1] if len(l5) > 1 else None,
            "week":    l5[2] if len(l5) > 2 else None,
            "L2Pflag": l5[3] if len(l5) > 3 else None,

            "SVaccuracy": l6[0] if len(l6) > 0 else None,
            "SVhealth":   l6[1] if len(l6) > 1 else None,
            "TGD":        l6[2] if len(l6) > 2 else None,
            "IODC":       l6[3] if len(l6) > 3 else None,

            "Tom":    l7[0] if len(l7) > 0 else None,
            "FitInt": l7[1] if len(l7) > 1 else None,
        }
        info["gps"] = gps
        info["toe_tb"] = gps["Toe"]

    elif consumed >= 4: 
        l1 = cols[0] if len(cols) > 0 else []
        l2 = cols[1] if len(cols) > 1 else []
        l3 = cols[2] if len(cols) > 2 else []
        l4 = cols[3] if len(cols) > 3 else []

        glo = {
            "tauN":   clock[0] if len(clock) > 0 else None,
            "gammaN": clock[1] if len(clock) > 1 else None,
            "tk":     clock[2] if len(clock) > 2 else None,

            "x":      l1[0] if len(l1) > 0 else None,
            "xDot":   l1[1] if len(l1) > 1 else None,
            "xAcc":   l1[2] if len(l1) > 2 else None,
            "health": l1[3] if len(l1) > 3 else None,

            "y":        l2[0] if len(l2) > 0 else None,
            "yDot":     l2[1] if len(l2) > 1 else None,
            "yAcc":     l2[2] if len(l2) > 2 else None,
            "freq_num": l2[3] if len(l2) > 3 else None,

            "z":    l3[0] if len(l3) > 0 else None,
            "zDot": l3[1] if len(l3) > 1 else None,
            "zAcc": l3[2] if len(l3) > 2 else None,
            "age":  l3[3] if len(l3) > 3 else None,
        }
        info["glo"] = glo
        info["toe_tb"] = glo["tk"]
    else:
        info["toe_tb"] = None
    return info, consumed

class Orbit:#сохроняем все в контейнер
    def __init__(self, d):
        self.system = d.get("system")
        self.prn = d.get("prn")
        self.epoch = d.get("epoch")
        self.toe_tb = d.get("toe_tb")
        self.raw = d.get("raw", [])
        self.gps = d.get("gps")
        self.glo = d.get("glo")
    def __getattr__(self, n):
        if self.gps and n in self.gps: return self.gps[n]
        if self.glo and n in self.glo: return self.glo[n]
        raise AttributeError(n)
    
records = []; all_epochs=set(); all_prns=set()
for fn in FILENAMES:
    if not Path(fn).exists(): 
        continue
    body = read_body(fn)
    i=0
    while i < len(body):
        tokens = body[i].strip().split()
        if not tokens: i+=1;continue
        if is_block_start(tokens):
            try: info, step = parse_block(body,i); records.append(info)
            except: step=1
            all_epochs.add(info["epoch"]); 
            all_prns.add(info["prn"]); 
            i+=step
        else: i+=1

epochs_sorted=sorted(all_epochs)
prns_sorted=sorted(all_prns,key=lambda x:(x[0],int(x[1:])))
params=[[None for _ in prns_sorted] for _ in epochs_sorted]
for rec in records:
    ei=epochs_sorted.index(rec["epoch"]); si=prns_sorted.index(rec["prn"])
    params[ei][si]=Orbit(rec)

def show_available_pairs():#вывод возможных пар
    print("Все возможные пары (из файлов):")
    valid_map = {}
    for i, ep in enumerate(epochs_sorted):
        js = [j for j in range(len(prns_sorted)) if params[i][j] is not None]
        if js:
            valid_map[i] = js
            js_str = " ".join(str(j) for j in js)
            sat_list = [prns_sorted[j] for j in js]
            sat_str = ", ".join(sat_list)
            print(f"[{i}] [{js_str}] {sat_str}")  
    return valid_map

def parse_pairs_input(s, valid_map):#проверка ввовода возможных пар
    raw_pairs = [(int(i), int(j)) for i, j in re.findall(r'(\d+)\s*,\s*(\d+)', s)]
    if not raw_pairs:
        print("Не удалось распознать ни одной пары ")
        return []

    pairs = []
    for i, j in raw_pairs:
        if i not in valid_map:
            print(f"— эпоха [{i}] отсутствует среди доступных.")
            continue
        if j not in valid_map[i]:
            print(f"— пара [{i},{j}] отсутствует среди доступных.")
            continue
        pairs.append((i, j))
    return pairs

def _fmt_num(v):
    if v is None: 
        return "—"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)

def _format_raw(raw, line_width=100):
    if not raw: return "[]"
    parts = [_fmt_num(x) for x in raw]
    lines = []
    cur = "["
    for p in parts:
        add = ("" if cur == "[" else ", ") + p
        if len(cur) + len(add) > line_width:
            lines.append(cur)
            cur = " " + p
        else:
            cur += add
    cur += "]"
    lines.append(cur)
    return "\n".join(lines)

def _print_table(rows, key_w=None, skip_none=True):
    if not rows: 
        return
    if skip_none:
        rows = [(k, v) for (k, v) in rows if v is not None]
    if not rows:
        return
    if key_w is None:
        key_w = max(12, max(len(k) for k, _ in rows))
    for k, v in rows:
        print(f"{k:<{key_w}} | {_fmt_num(v)}")

def print_raw_then_detailed_for(i, j, skip_none=True):
    o = params[i][j]
    header = f"[{i:3d}][{j:3d}]  {epochs_sorted[i].isoformat()}  {prns_sorted[j]} ({o.system})"
    print()
    print("=" * 100)
    print(header)
    print("=" * 100)
    print("Эфемерид:")
    print(_format_raw(o.raw))
    print("Подробный (переменная | значение ):")
    common = [
        ("system", o.system),
        ("prn", o.prn),
        ("epoch", epochs_sorted[i].isoformat()),
        ("toe_tb", o.toe_tb),
    ]
    _print_table(common, key_w=10, skip_none=False)
    if o.system == "G":
        keys = ["af0","af1","af2",
                 "IODE","Crs","dn","M0",
                 "Cuc","e","Cus","sqrtA",
                 "Toe","Cic","Omega0","Cis",
                 "i0","Crc","omega","OmegaDot",
                 "IDOT","codesL2","week","L2Pflag",
                 "SVaccuracy","SVhealth","TGD","IODC",
                 "Tom","FitInt"]
        rows = [(k, getattr(o, k, o.gps.get(k) if o.gps else None)) for k in keys]
        _print_table(rows, key_w=10, skip_none=skip_none)
    elif o.system == "R":
        keys = [ "tauN","gammaN","tk",
                 "x","xDot","xAcc","health",
                 "y","yDot","yAcc","freq_num",
                 "z","zDot","zAcc","age"]
        rows = [(k, getattr(o, k, o.glo.get(k) if o.glo else None)) for k in keys]
        _print_table(rows, key_w=10, skip_none=skip_none)
    
def calc_orbit_print(i, j, o):
    prn_short = f"{o.system}{o.prn[-2:]}" if hasattr(o, "prn") else f"{o.system}"
    print("-" * 100)
    print(f"Калькуляция: [{i}][{j}]   {o.epoch.isoformat()}   {prn_short} ({o.system})")
    print("-" * 100)

    if o.system == "G" and getattr(o, "gps", None):
        g = o.gps
        a  = (g.get("sqrtA") or 0.0) ** 2
        e  =  g.get("e")     or 0.0
        M0 =  g.get("M0")    or 0.0
        print(f"a = {a:,.3f} м;   e = {e:.8f};   M0 = {M0:.6f} рад")

    elif o.system == "R" and getattr(o, "glo", None):
        r = o.glo
        X  = (r.get("x")    or 0.0)  
        Y  = (r.get("y")    or 0.0)  
        Z  = (r.get("z")    or 0.0)  
        Xd = (r.get("xDot") or 0.0)  
        Yd = (r.get("yDot") or 0.0)  
        Zd = (r.get("zDot") or 0.0)  
        tk = (r.get("tk")   or 0)
        print(f"X={X:,.1f} км;  Y={Y:,.1f} км;  Z={Z:,.1f} км;   tk={int(tk)} с")
        print(f"Xdot={Xd:.3f} км/с;  Ydot={Yd:.3f} км/с;  Zdot={Zd:.3f} км/с")

def main():
    valid_map = show_available_pairs()
    print("Нужно выбрать пару.")
    print("Пример: i,j")
    user_in = input("Ввод: ").strip()
    pairs = parse_pairs_input(user_in, valid_map)
    if not pairs:
        return

    print("Режим вывода:")
    print("1 — вывести RAW и подробный блоки")
    print("2 — не выводить блоки")
    mode = input("Выбери 1 или 2: ").strip()
    show_blocks = (mode == "1")
    for i, j in pairs:
        if show_blocks:                              
            print_raw_then_detailed_for(i, j)
       
        calc_orbit_print(i, j, params[i][j])

if __name__ == "__main__":
    main()
