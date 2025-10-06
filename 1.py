# 1.py — читает (GPS + GLONASS), показывает меню выбора
# и возвращает (sv_code, epoch_iso, block_lines, system_name) .

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Iterable, Optional
from collections import OrderedDict
import re

# Пути к файлам 
GPS_PATH = Path("BRDC0010.17n")
GLO_PATH = Path("base_01.rnx")

# поиск эрохи, тоесть пробегает по первым строкам
_FIRSTLINE_RE = re.compile(
    r"^\s*(\d{1,2})\s+(\d{2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})"
)

# пропускаем заголовок
def _read_after_header(fp) -> List[str]:
    lines: List[str] = []
    in_header = True
    for line in fp:
        if in_header:
            if "END OF HEADER" in line:
                in_header = False
            continue
        lines.append(line.rstrip("\n"))
    return lines

# разделяем весь файл на отдельные блоки эфемерид
def _split_blocks(lines: Iterable[str]) -> List[List[str]]:
    blocks: List[List[str]] = []
    cur: List[str] = []
    for ln in lines:
        if _FIRSTLINE_RE.match(ln):
            if cur:
                blocks.append(cur)
            cur = [ln]
        else:
            if cur:
                cur.append(ln)
    if cur:
        blocks.append(cur)
    return blocks

# анализатор заголовка одного блока эфемерид
def _parse_block_header(block: List[str]) -> Tuple[int, datetime]:
    m = _FIRSTLINE_RE.match(block[0])
    if not m:
        raise ValueError("Invalid RINEX NAV block header.")
    id_, yy, mo, dd, hh, mm, ss = map(int, m.groups())
    year = 1900 + yy if yy >= 80 else 2000 + yy
    return id_, datetime(year, mo, dd, hh, mm, ss)

# собираем все в одно целое
def _read_nav_file(path: Path, sys_char: str) -> List[Tuple[str, datetime, List[str]]]:
    """Вернёт список кортежей: (sv_code, epoch, block_lines)"""
    if not path.exists():
        return []
    with path.open("r", errors="ignore") as fp:
        lines = _read_after_header(fp)
    blocks = _split_blocks(lines)
    out: List[Tuple[str, datetime, List[str]]] = []
    for b in blocks:
        prn, epoch = _parse_block_header(b)
        sv_code = f"{sys_char}{prn:02d}"
        out.append((sv_code, epoch, b))
    return out

# Глобальные индексы для простоты: epoch_iso -> {sv_code -> block_lines}
svInfo: Dict[str, Dict[str, Dict[str, object]]] = OrderedDict()
# преобразуем для выбора НКА
def _build_indexes() -> None:
    """Заполняет svInfo: epoch_iso -> {sv_code: {'block': [...], 'system': 'GPS'/'GLONASS'}}"""
    all_entries: List[Tuple[str, datetime, List[str], str]] = []
    for sv_code, epoch, lines in _read_nav_file(GPS_PATH, "G"):
        all_entries.append((sv_code, epoch, lines, "GPS"))
    for sv_code, epoch, lines in _read_nav_file(GLO_PATH, "R"):
        all_entries.append((sv_code, epoch, lines, "GLONASS"))

    # сгруппируем по эпохам
    tmp: Dict[str, Dict[str, Dict[str, object]]] = {}
    for sv_code, epoch, lines, system in all_entries:
        ek = epoch.isoformat()
        tmp.setdefault(ek, {})[sv_code] = {"block": lines, "system": system}

    # отсортируем по времени
    for ek in sorted(tmp.keys(), key=lambda s: datetime.fromisoformat(s)):
        svInfo[ek] = tmp[ek]

# список всех НКА
def _all_sv_sorted() -> List[str]:
    s = set()
    for ek, m in svInfo.items():
        s.update(m.keys())
    return sorted(s)

def print_sv_menu() -> List[str]:
    all_sv = _all_sv_sorted()
    gps = [sv for sv in all_sv if sv.startswith("G")]
    glo = [sv for sv in all_sv if sv.startswith("R")]
    if gps:
        print(" GPS:", " ".join(gps))
    if glo:
        print(" GLO:", " ".join(glo))
    print("\nСписок для выбора (номер — НКА — количество эпох):")
    for i, sv in enumerate(all_sv, 1):
        cnt = sum(1 for ek in svInfo if sv in svInfo[ek])
        print(f"[{i:02d}] {sv}  —  {cnt} эпох")
    return all_sv

def print_epoch_menu_for_sv(sv_code: str) -> List[str]:
    eps = [ek for ek in svInfo.keys() if sv_code in svInfo[ek]]
    print(f"\nДоступные эпохи для {sv_code}:")
    for i, ek in enumerate(eps, 1):
        print(f"[{i:02d}] {ek}")
    return eps

def select_block_interactive():
    """
    Интерактивно выбрать НКА и эпоху и ВЕРНУТЬ:
      sv_code: 'G01' / 'R03'
      epoch_key: 'YYYY-MM-DDTHH:MM:SS'
      block_lines: list[str]
      system_name: 'GPS' или 'GLONASS'
    """
    if not svInfo:
        _build_indexes()

    all_sv = print_sv_menu()
    while True:
        raw_sv_idx = input("\nВведи номер НКА (Enter — выход): ").strip()
        if raw_sv_idx == "":
            return None
        if not raw_sv_idx.isdigit():
            print("Ожидался номер из списка НКА."); continue
        sv_idx = int(raw_sv_idx)
        if not (1 <= sv_idx <= len(all_sv)):
            print("Нет такого номера НКА."); continue

        sv_code = all_sv[sv_idx - 1]
        eps_for_sv = print_epoch_menu_for_sv(sv_code)
        if not eps_for_sv:
            print("Для выбранного НКА нет доступных эпох."); continue

        raw_ep_idx = input("\nВведи номер эпохи (Enter — назад): ").strip()
        if raw_ep_idx == "":
            continue
        if not raw_ep_idx.isdigit():
            print("Ожидался номер эпохи из списка."); continue
        ep_idx = int(raw_ep_idx)
        if not (1 <= ep_idx <= len(eps_for_sv)):
            print("Нет такого номера эпохи."); continue

        epoch_key = eps_for_sv[ep_idx - 1]
        rec = svInfo[epoch_key].get(sv_code)
        if rec is None:
            print("Внутренняя ошибка: блок не найден."); continue

        show_raw = input("Показать «сырой» блок эфемерид? (y/N): ").strip().lower()
        if show_raw == "y":
            print("--- Сырой блок ---")
            for line in rec["block"]:
                print(line)
            print("--- Конец блока ---")

        return sv_code, epoch_key, rec["block"], rec["system"]

if __name__ == "__main__":
    pick = select_block_interactive()
    if pick is None:
        print("Выход.")
    else:
        sv_code, epoch_key, block_lines, system_name = pick
        print(f"\nВыбрано: {sv_code} @ {epoch_key} ({system_name}), строк в блоке: {len(block_lines)}")
