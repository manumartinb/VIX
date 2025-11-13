# -*- coding: utf-8 -*-
r"""
V1 VIX Calendar Forward Factor Scanner - Escanea calendarios VIX y calcula Forward Factor

OBJETIVO: Escanea TODOS los calendarios posibles (CALLs y PUTs) en archivos 30MIN, calcula FF (backwardation/contango)

OUTPUT CSV: date_us, date_es, file, right, DTE1, DTE2, expiration1, expiration2, strike, SPOT,
            DELTA_K1, DELTA_K2, IV1, IV2, T1_years, T2_years, Forward_Factor, Forward_Factor_x1000,
            SPREAD, url

FILTROS (activables con ENABLE_XXX_FILTER):
1. DELTA_K1: filtra por |delta| del strike corto K1 (funciona para CALLs y PUTs)
2. DTE_DIFF_PCT: limita diferencia porcentual entre DTEs [(DTE2-DTE1)/DTE1]
3. DTE_DIFF: limita diferencia absoluta entre DTEs en días (DTE2-DTE1)
4. MAX_DTE1: limita el DTE máximo permitido para el primer vencimiento
5. SPREAD: limita spread bid-ask relativo (máximo de ambas piernas del calendar)
"""

# ================== IMPORTS ==================
import re
import random
import math
import os
from pathlib import Path
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from collections import Counter

import numpy as np
import pandas as pd

# ================== RUTAS ==================
DATA_DIR = r"C:\Users\Administrator\Desktop\FINAL DATA\HIST AND STREAMING DATA\STREAMING\VIX"
DESKTOP = Path.home() / "Desktop"

# ================== ZONAS HORARIAS ==================
TZ_US = ZoneInfo("America/New_York")
TZ_ES = ZoneInfo("Europe/Madrid")

# ================== CONFIG SNAPSHOT ==================
TARGET_HHMMS_US = ["10:30"]               # Lista de horas US objetivo para snapshot (formato "HH:MM")
NEAREST_MINUTE_TOLERANCE = 0              # Tolerancia en minutos para encontrar timestamp más cercano
IGNORE_TARGET_MINUTE = True              # Control de procesamiento completo del día
                                           # False: solo procesa los timestamps en TARGET_HHMMS_US
                                           # True: ignora TARGET_HHMMS_US y procesa TODOS los timestamps del CSV
                                           # ATENCIÓN: True puede generar miles de estructuras por día

# ========== CONFIGURACION ==========
FILENAME_TO_PROCESS = "SNAP*.csv"  # Pattern de archivos a procesar
NUM_RANDOM_FILES = 1  # Solo si FILENAME_TO_PROCESS = None

# Filtro Delta K1 (absoluto, funciona para CALLs y PUTs)
ENABLE_DELTA_K1_FILTER = True
DELTA_K1_MIN = 0.2  # Mínimo |delta| del strike corto K1 (None = sin límite)
DELTA_K1_MAX = 0.8  # Máximo |delta| del strike corto K1 (None = sin límite)

# Rangos DTE
RANGE_A = (1, 9999)  # DTE front leg
RANGE_B = (1, 9999)  # DTE back leg

# Filtro DTE_DIFF_PCT: limita diferencia porcentual entre DTEs
ENABLE_DTE_DIFF_PCT_FILTER = True
MAX_DTE_DIFF_PCT = 10.0  # ! DTE_DIFF_PCT = (DTE2 - DTE1) / DTE1. 10 = 1000% (Así NO excluye calendars tipo 1-4 o 4-7)

# Filtro DTE_DIFF: limita diferencia absoluta entre DTEs en días
ENABLE_DTE_DIFF_FILTER = True
MAX_DTE_DIFF = 50  # ! Máximo DTE2 - DTE1 en días (None = sin límite)

# Filtro DTE1: limita el DTE máximo permitido para el primer vencimiento
ENABLE_MAX_DTE1_FILTER = False
MAX_DTE1 = 9999  # Máximo DTE permitido para el primer vencimiento (None = sin límite)

# Filtro SPREAD: limita spread bid-ask relativo (control de calidad de precios)
ENABLE_SPREAD_FILTER = False
MAX_SPREAD_REL = 9999  # Spread bid-ask máximo relativo permitido (0.5 = 50% del mid)
                      # Descarta opciones con spread muy amplio (ilíquidas o precios sospechosos)
                      # EJEMPLO: Si mid=10, bid=7, ask=13, spread=6, spread_rel=60% → RECHAZA

# ================== FUNCIONES AUXILIARES ==================
BASE_URL_OS = "https://optionstrat.com/build/custom/VIX/"

def safe_filename(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', '-', text)

def is_third_friday(d):
    import calendar as cal
    calm = cal.monthcalendar(d.year, d.month)
    fridays = [wk[cal.FRIDAY] for wk in calm if wk[cal.FRIDAY]!=0]
    return len(fridays)>=3 and d.day==fridays[2]

def root_for_exp(exp_str):
    """Determina el root para VIX (siempre VIX)"""
    return "VIX"

def yyyymmdd_to_yymmdd(d):
    """Convierte fecha a formato YYMMDD para URL"""
    return d.strftime("%y%m%d")

def make_url_optionstrat(right, exp1, k1, exp2, k2):
    """
    Genera URL de Optionstrat para un calendar spread VIX.

    Args:
        right: 'C' o 'P' (o 'CALL'/'PUT')
        exp1: Expiración front leg (formato "YYYY-MM-DD")
        k1: Strike front leg
        exp2: Expiración back leg (formato "YYYY-MM-DD")
        k2: Strike back leg

    Returns:
        str: URL de Optionstrat

    Ejemplo:
        https://optionstrat.com/build/custom/VIX/.VIX240315C20x-1,.VIX240415C20x1
    """
    # Convertir expiraciones a datetime
    date1 = datetime.strptime(exp1, "%Y-%m-%d")
    date2 = datetime.strptime(exp2, "%Y-%m-%d")

    # Obtener formato YYMMDD
    ymd1 = yyyymmdd_to_yymmdd(date1)
    ymd2 = yyyymmdd_to_yymmdd(date2)

    # Determinar roots (siempre VIX)
    r1 = root_for_exp(exp1)
    r2 = root_for_exp(exp2)

    # Normalizar right a 'C' o 'P'
    right_upper = str(right).upper().strip()
    if right_upper in ['C', 'CALL']:
        letter = 'C'
    elif right_upper in ['P', 'PUT']:
        letter = 'P'
    else:
        letter = 'C'  # Default

    # Construir legs
    # Front leg: short (x-1)
    # Back leg: long (x1)
    leg1 = f".{r1}{ymd1}{letter}{int(k1)}x-1"
    leg2 = f".{r2}{ymd2}{letter}{int(k2)}x1"

    # Construir URL completa
    url = BASE_URL_OS + ",".join([leg1, leg2])

    return url

def make_url_optionstrat_with_roots(right, exp1, k1, exp2, k2, root1, root2):
    """
    Genera URL de Optionstrat para un calendar spread usando roots específicos.

    Args:
        right: 'C' o 'P' (o 'CALL'/'PUT')
        exp1: Expiración front leg (formato "YYYY-MM-DD")
        k1: Strike front leg
        exp2: Expiración back leg (formato "YYYY-MM-DD")
        k2: Strike back leg
        root1: Root symbol del front leg (VIX)
        root2: Root symbol del back leg (VIX)

    Returns:
        str: URL de Optionstrat

    Ejemplo:
        https://optionstrat.com/build/custom/VIX/.VIX240315C20x-1,.VIX240415C20x1
    """
    # Convertir expiraciones a datetime
    date1 = datetime.strptime(exp1, "%Y-%m-%d")
    date2 = datetime.strptime(exp2, "%Y-%m-%d")

    # Obtener formato YYMMDD
    ymd1 = yyyymmdd_to_yymmdd(date1)
    ymd2 = yyyymmdd_to_yymmdd(date2)

    # Normalizar right a 'C' o 'P'
    right_upper = str(right).upper().strip()
    if right_upper in ['C', 'CALL']:
        letter = 'C'
    elif right_upper in ['P', 'PUT']:
        letter = 'P'
    else:
        letter = 'C'  # Default

    # Construir legs (front: short x-1, back: long x1)
    leg1 = f".{root1}{ymd1}{letter}{int(k1)}x-1"
    leg2 = f".{root2}{ymd2}{letter}{int(k2)}x1"
    url = BASE_URL_OS + ",".join([leg1, leg2])
    return url

def list_local_files(data_dir):
    """Lista archivos 30MINDATA_*.csv en orden cronológico"""
    pattern = re.compile(r"^30MINDATA[_-](\d{4})-?(\d{2})-?(\d{2})\.csv$", re.IGNORECASE)
    files_found = []
    for fname in os.listdir(data_dir):
        m = pattern.match(fname)
        if m:
            year, month, day = m.groups()
            try:
                dt = datetime(int(year), int(month), int(day))
                files_found.append((dt, fname))
            except ValueError:
                continue
    files_found.sort(key=lambda x: x[0])
    return [f for _, f in files_found]

def parse_timestamp_us(ts_val):
    """Parsea timestamp a datetime US"""
    if pd.isna(ts_val):
        return None
    if isinstance(ts_val, (datetime, pd.Timestamp)):
        dt = pd.Timestamp(ts_val)
        if dt.tzinfo is None:
            dt = dt.tz_localize(TZ_US)
        else:
            dt = dt.tz_convert(TZ_US)
        return dt
    s = str(ts_val).strip()
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"]:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=TZ_US)
        except ValueError:
            continue
    return None

def find_nearest_snapshot(df_file, target_times_hhmm, ignore_target_minute=False):
    """Encuentra snapshot más cercano a horas objetivo. Si ignore_target_minute=True, devuelve TODO el DataFrame"""
    if "ms_of_day" not in df_file.columns:
        return None

    if ignore_target_minute:
        return df_file.copy()

    # Convertir ms_of_day a string y parsear como tiempo
    # El formato puede ser "HH:MM:SS.sss" o milisegundos numéricos
    def parse_time_to_ms(val):
        """Convierte ms_of_day a milisegundos desde medianoche"""
        if pd.isna(val):
            return None

        # Intentar convertir a numérico (ya está en milisegundos)
        try:
            ms = float(val)
            if np.isfinite(ms):
                return ms
        except (ValueError, TypeError):
            pass

        # Intentar parsear como string de tiempo HH:MM:SS.sss
        try:
            time_str = str(val).strip()
            # Formato: "10:30:00.000" o "10:30:00"
            parts = time_str.split(':')
            if len(parts) >= 2:
                hh = int(parts[0])
                mm = int(parts[1])
                ss = 0
                ms_frac = 0

                if len(parts) >= 3:
                    # Puede tener segundos y milisegundos
                    ss_parts = parts[2].split('.')
                    ss = int(ss_parts[0])
                    if len(ss_parts) > 1:
                        ms_frac = int(ss_parts[1])

                # Convertir todo a milisegundos
                total_ms = hh * 3600_000 + mm * 60_000 + ss * 1000 + ms_frac
                return total_ms
        except (ValueError, IndexError, AttributeError):
            pass

        return None

    # Aplicar conversión a TODO el dataframe
    df_file_copy = df_file.copy()
    df_file_copy["ms_of_day_numeric"] = df_file_copy["ms_of_day"].apply(parse_time_to_ms)
    df_file_copy = df_file_copy.dropna(subset=["ms_of_day_numeric"]).copy()

    if df_file_copy.empty:
        return None

    # Obtener valores únicos de ms_of_day para identificar snapshots
    unique_times = df_file_copy["ms_of_day_numeric"].unique()

    target_ms_list = []
    for hhmm_str in target_times_hhmm:
        hh, mm = map(int, hhmm_str.split(":"))
        target_ms_list.append(hh * 3600_000 + mm * 60_000)

    best_time_ms = None
    best_diff = None

    # Encontrar el snapshot (tiempo) más cercano
    for tms in target_ms_list:
        for unique_time in unique_times:
            diff_val = abs(unique_time - tms)
            if diff_val <= NEAREST_MINUTE_TOLERANCE * 60_000:
                if best_diff is None or diff_val < best_diff:
                    best_diff = diff_val
                    best_time_ms = unique_time

    # Devolver TODAS las filas de ese snapshot
    if best_time_ms is not None:
        mask = df_file_copy["ms_of_day_numeric"] == best_time_ms
        result = df_file_copy[mask].copy()
        # Eliminar la columna temporal
        result = result.drop(columns=["ms_of_day_numeric"], errors='ignore')
        return result

    return None

def filter_df_by_root(df_sub, desired_root):
    """
    Filtra DataFrame por root symbol (VIX).

    IMPORTANTE: Esta función es PERMISIVA. Si la columna root no existe o tiene valores
    vacíos/NULL, NO descarta las filas. Solo filtra cuando hay valores explícitos que
    no coinciden.
    """
    if desired_root is None:
        return df_sub

    desired_root_upper = desired_root.strip().upper()

    # Buscar columna root
    root_col = None
    if "root" in df_sub.columns:
        root_col = "root"
    else:
        for col_candidate in ["root_symbol", "rootSymbol", "Root", "ROOT"]:
            if col_candidate in df_sub.columns:
                root_col = col_candidate
                break

    # Si NO hay columna root, devolver todo sin filtrar
    if root_col is None:
        return df_sub

    # Filtrar SOLO las filas que tienen un valor explícito que coincide con desired_root
    # O que tienen valores vacíos/NULL (ser permisivo)
    df_sub_copy = df_sub.copy()
    df_sub_copy['_root_value'] = df_sub_copy[root_col].astype(str).str.strip().str.upper()

    # Incluir:
    # 1. Filas donde root coincide con desired_root
    # 2. Filas donde root es vacío, NULL, "NAN", etc.
    mask = (
        (df_sub_copy['_root_value'] == desired_root_upper) |
        (df_sub_copy[root_col].isna()) |
        (df_sub_copy['_root_value'] == '') |
        (df_sub_copy['_root_value'] == 'NAN') |
        (df_sub_copy['_root_value'] == 'NONE')
    )

    result = df_sub_copy[mask].drop(columns=['_root_value']).copy()

    # Logging de diagnóstico
    filtered_out = len(df_sub) - len(result)
    if filtered_out > 0:
        # Mostrar qué valores se descartaron
        discarded_roots = df_sub_copy[~mask]['_root_value'].unique()
        print(f"        [DEBUG] filter_df_by_root: descartadas {filtered_out} filas con root={list(discarded_roots)} (esperaba {desired_root_upper})")

    return result

def get_mid_from_row(row):
    """Obtiene precio mid de una fila"""
    try:
        bid_val = row.get("bid") if hasattr(row, "get") else row["bid"]
        ask_val = row.get("ask") if hasattr(row, "get") else row["ask"]

        bid_num = float(bid_val) if pd.notna(bid_val) else None
        ask_num = float(ask_val) if pd.notna(ask_val) else None

        if bid_num is not None and ask_num is not None:
            return (bid_num + ask_num) / 2.0
        return None
    except (KeyError, TypeError, ValueError):
        return None

def get_iv_from_row(row):
    """Obtiene IV de una fila"""
    for col in ["implied_vol", "iv", "implied_volatility", "IV", "ImpliedVolatility", "IV_BS"]:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                try:
                    iv_val = float(val)
                    if np.isfinite(iv_val) and iv_val > 0:
                        return iv_val
                except (TypeError, ValueError):
                    continue
    return None

def get_delta_from_row(row):
    """Obtiene delta de una fila"""
    for col in ["delta", "Delta", "DELTA", "delta_bs", "Delta_BS"]:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                try:
                    delta_val = float(val)
                    if np.isfinite(delta_val):
                        return delta_val
                except (TypeError, ValueError):
                    continue
    return None

def get_spread_rel_from_row(row):
    """
    Calcula spread relativo (bid-ask spread / mid) de una fila.
    Returns: (spread_rel, bid, ask, mid) o (None, None, None, None) si no se puede calcular
    """
    try:
        bid_val = row.get("bid") if hasattr(row, "get") else row["bid"]
        ask_val = row.get("ask") if hasattr(row, "get") else row["ask"]

        bid = float(bid_val) if pd.notna(bid_val) else None
        ask = float(ask_val) if pd.notna(ask_val) else None

        if bid is None or ask is None:
            return None, None, None, None

        if not (np.isfinite(bid) and np.isfinite(ask)):
            return None, None, None, None

        if bid <= 0 or ask <= 0:
            return None, None, None, None

        if ask < bid:
            return None, None, None, None

        mid = 0.5 * (bid + ask)

        if not np.isfinite(mid) or mid <= 0:
            return None, None, None, None

        spread_rel = (ask - bid) / mid

        if not np.isfinite(spread_rel):
            return None, None, None, None

        return spread_rel, bid, ask, mid

    except (KeyError, TypeError, ValueError):
        return None, None, None, None

def leg_quote_ok(row, max_spread_rel=MAX_SPREAD_REL):
    """
    Valida que una opción tenga un spread bid-ask aceptable.

    Args:
        row: Fila del DataFrame con datos de la opción
        max_spread_rel: Spread máximo relativo permitido (default: MAX_SPREAD_REL)

    Returns:
        bool: True si el spread es aceptable, False en caso contrario
    """
    spread_rel, _, _, _ = get_spread_rel_from_row(row)

    if spread_rel is None:
        return False

    if spread_rel > float(max_spread_rel):
        return False

    return True

def compute_forward_factor(iv1, iv2, t1_years, t2_years):
    """Calcula FF comparando IV front con IV forward. FF>0:backwardation, FF<0:contango. Formula: FF = IV1/IV_fwd - 1"""
    try:
        if iv1 is None or iv2 is None or t1_years is None or t2_years is None:
            return np.nan

        iv1_f = float(iv1)
        iv2_f = float(iv2)
        t1 = float(t1_years)
        t2 = float(t2_years)
        if not (np.isfinite(iv1_f) and np.isfinite(iv2_f) and np.isfinite(t1) and np.isfinite(t2)):
            return np.nan

        if iv1_f <= 0 or iv2_f <= 0 or t1 <= 0 or t2 <= 0:
            return np.nan

        # Validar T2 > T1
        if t2 <= t1:
            return np.nan

        # Calcular varianzas
        v1 = iv1_f * iv1_f
        v2 = iv2_f * iv2_f

        # Calcular varianza forward
        v_fwd = (v2 * t2 - v1 * t1) / (t2 - t1)

        # Validar varianza forward positiva
        if v_fwd <= 0:
            return np.nan

        # Calcular IV forward
        iv_fwd = math.sqrt(v_fwd)

        # Calcular Forward Factor
        ff = (iv1_f / iv_fwd) - 1.0

        return round(ff, 6)

    except Exception:
        return np.nan

def process_one_30min_file(filename, data_dir):
    """Procesa archivo 30MIN y extrae calendarios. Returns: list of dict"""
    filepath = os.path.join(data_dir, filename)

    # Extraer fecha del nombre (soporta múltiples formatos)
    date_obj = None

    # Formato 1: 30MINDATA_2025-10-24.csv o 30MINDATA-20251024.csv
    pattern1 = re.compile(r"^30MINDATA[_-](\d{4})-?(\d{2})-?(\d{2})\.csv$", re.IGNORECASE)
    m1 = pattern1.match(filename)
    if m1:
        year, month, day = m1.groups()
        try:
            date_obj = datetime(int(year), int(month), int(day)).date()
        except ValueError:
            pass

    # Formato 2: SNAPSHOT_CHAIN_VIX_2025-10-31_1555_ET.csv
    if date_obj is None:
        pattern2 = re.compile(r"SNAPSHOT.*?(\d{4})-(\d{2})-(\d{2})", re.IGNORECASE)
        m2 = pattern2.search(filename)
        if m2:
            year, month, day = m2.groups()
            try:
                date_obj = datetime(int(year), int(month), int(day)).date()
            except ValueError:
                pass

    # Formato 3: Cualquier CSV con fecha YYYY-MM-DD
    if date_obj is None:
        pattern3 = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
        m3 = pattern3.search(filename)
        if m3:
            year, month, day = m3.groups()
            try:
                date_obj = datetime(int(year), int(month), int(day)).date()
            except ValueError:
                pass

    # Si no se pudo extraer fecha, leerla del CSV
    if date_obj is None:
        print(f"\n{'='*70}")
        print(f"[ARCHIVO] {filename}")
        print(f"  [!] No se pudo extraer fecha del nombre, intentando leer del CSV...")
        try:
            df_temp = pd.read_csv(filepath, nrows=1, low_memory=False)
            if "date" in df_temp.columns:
                date_str = str(df_temp["date"].iloc[0])
                date_obj = pd.to_datetime(date_str).date()
                print(f"  [OK] Fecha extraída del CSV: {date_obj}")
            else:
                print(f"  [X] No se encontró columna 'date' en el CSV")
                return []
        except Exception as e:
            print(f"  [X] Error leyendo CSV para extraer fecha: {e}")
            return []

    date_us_obj = datetime.combine(date_obj, datetime.min.time()).replace(tzinfo=TZ_US)
    date_us_str = date_us_obj.strftime("%Y-%m-%d")
    date_es_obj = date_us_obj.astimezone(TZ_ES)
    date_es_str = date_es_obj.strftime("%Y-%m-%d")

    print(f"\n{'='*70}")
    print(f"[ARCHIVO] {filename}")
    print(f"  Fecha US: {date_us_str} | Fecha ES: {date_es_str}")
    print(f"{'='*70}")

    # Cargar archivo
    try:
        df_file = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"  [X] Error leyendo archivo: {e}")
        return []

    print(f"  [OK] Archivo cargado: {len(df_file)} filas")

    # Encontrar snapshot
    if IGNORE_TARGET_MINUTE:
        print(f"  [INFO] IGNORE_TARGET_MINUTE = True -> Procesando TODO el archivo (todos los timestamps)")
    else:
        print(f"  [INFO] IGNORE_TARGET_MINUTE = False -> Buscando snapshot en {TARGET_HHMMS_US}")

    df_snap = find_nearest_snapshot(df_file, TARGET_HHMMS_US, ignore_target_minute=IGNORE_TARGET_MINUTE)
    if df_snap is None or df_snap.empty:
        print(f"  [X] No se encontró snapshot válido")
        return []

    if IGNORE_TARGET_MINUTE:
        print(f"  [OK] Procesando {len(df_snap)} opciones (todos los timestamps del día)")
    else:
        print(f"  [OK] Snapshot encontrado: {len(df_snap)} opciones")

    # Validar columnas requeridas
    required_cols = ["right", "expiration", "strike"]
    if not all(col in df_snap.columns for col in required_cols):
        print(f"  [X] Faltan columnas requeridas")
        return []

    # Obtener SPOT (intenta múltiples columnas candidatas)
    spot = None
    spot_column_candidates = ["underlying_price", "underlying_last", "spot", "S", "underlying"]

    for col_name in spot_column_candidates:
        if col_name in df_snap.columns:
            spot_values = pd.to_numeric(df_snap[col_name], errors="coerce").dropna()
            if not spot_values.empty:
                spot = float(spot_values.iloc[0])
                if np.isfinite(spot) and spot > 0:
                    break

    if spot is None or not np.isfinite(spot) or spot <= 0:
        print(f"  [X] No se pudo obtener SPOT válido")
        print(f"      Columnas disponibles: {list(df_snap.columns[:10])}")
        return []

    spot = float(spot)
    print(f"  [OK] SPOT: {spot:.2f}")

    # Detectar formato de columna 'right' (puede ser 'C'/'P' o 'CALL'/'PUT')
    unique_rights = df_snap["right"].dropna().unique()
    if len(unique_rights) > 0:
        sample_right = str(unique_rights[0]).strip().upper()
        if sample_right in ['C', 'P']:
            call_value = 'C'
            put_value = 'P'
        else:
            call_value = 'CALL'
            put_value = 'PUT'
    else:
        call_value = 'CALL'
        put_value = 'PUT'

    # Siempre procesar ambos tipos (CALLs y PUTs)
    rights_to_process = [call_value, put_value]
    print(f"  [INFO] Procesando ambos tipos: {call_value} y {put_value}")

    # Convertir tipos comunes primero
    df_snap["strike"] = pd.to_numeric(df_snap["strike"], errors="coerce")
    df_snap["expiration"] = df_snap["expiration"].astype(str)

    # Calcular DTE
    # exp_date es naive (sin timezone), usamos solo la fecha de date_us_obj
    df_snap["exp_date"] = pd.to_datetime(df_snap["expiration"], errors="coerce")
    df_snap["DTE"] = (df_snap["exp_date"] - pd.Timestamp(date_obj)).dt.days

    # Acumulador de resultados
    results = []

    # Procesar cada tipo de opción
    for current_right in rights_to_process:
        print(f"\n  [PROCESANDO] Tipo: {current_right}")

        # Filtrar por RIGHT actual
        df_snap_filtered = df_snap[df_snap["right"].str.upper() == current_right.upper()].copy()

        if df_snap_filtered.empty:
            print(f"    [X] No hay opciones {current_right}")
            continue

        print(f"    [OK] Opciones {current_right}: {len(df_snap_filtered)}")

        # Filtrar por rangos DTE
        df_front = df_snap_filtered[(df_snap_filtered["DTE"] >= RANGE_A[0]) & (df_snap_filtered["DTE"] <= RANGE_A[1])].copy()
        df_back = df_snap_filtered[(df_snap_filtered["DTE"] >= RANGE_B[0]) & (df_snap_filtered["DTE"] <= RANGE_B[1])].copy()

        if df_front.empty or df_back.empty:
            print(f"    [X] No hay suficientes expiraciones en rangos DTE")
            print(f"        Front ({RANGE_A[0]}-{RANGE_A[1]}): {len(df_front)} opciones")
            print(f"        Back ({RANGE_B[0]}-{RANGE_B[1]}): {len(df_back)} opciones")
            continue

        print(f"    [OK] Front leg ({RANGE_A[0]}-{RANGE_A[1]} DTE): {len(df_front)} opciones")
        print(f"    [OK] Back leg ({RANGE_B[0]}-{RANGE_B[1]} DTE): {len(df_back)} opciones")

        # Obtener expiraciones únicas
        exps_front = sorted(df_front["expiration"].unique())
        exps_back = sorted(df_back["expiration"].unique())

        print(f"    [INFO] Expiraciones front: {len(exps_front)}")
        print(f"    [INFO] Expiraciones back: {len(exps_back)}")

        # Generar todos los calendarios para este tipo
        total_combos = 0
        total_exp_pairs_processed = 0
        total_exp_pairs_skipped_no_common_strikes = 0

        for exp1 in exps_front:
            for exp2 in exps_back:
                # Validar que exp2 > exp1
                try:
                    exp1_date = datetime.strptime(exp1, "%Y-%m-%d").date()
                    exp2_date = datetime.strptime(exp2, "%Y-%m-%d").date()
                except ValueError:
                    continue

                if exp2_date <= exp1_date:
                    continue

                total_exp_pairs_processed += 1

                # Obtener DTEs
                dte1 = (exp1_date - date_obj).days
                dte2 = (exp2_date - date_obj).days
                dte_diff = dte2 - dte1

                # FILTRO 0: MAX_DTE1 (DTE máximo del primer vencimiento)
                if ENABLE_MAX_DTE1_FILTER and MAX_DTE1 is not None:
                    if dte1 > MAX_DTE1:
                        continue

                # FILTRO 1: DTE_DIFF (diferencia absoluta en días)
                if ENABLE_DTE_DIFF_FILTER and MAX_DTE_DIFF is not None:
                    if dte_diff > MAX_DTE_DIFF:
                        continue

                # Calcular DTE_DIFF_PCT
                dte_diff_pct = (dte2 - dte1) / dte1 if dte1 > 0 else np.nan

                # FILTRO 2: DTE_DIFF_PCT (diferencia porcentual)
                if ENABLE_DTE_DIFF_PCT_FILTER and np.isfinite(dte_diff_pct):
                    if dte_diff_pct > MAX_DTE_DIFF_PCT:
                        continue

                # Calcular tiempos en años
                t1_years = dte1 / 365.0
                t2_years = dte2 / 365.0

                # Filtrar por expiración (no por root, para evitar descartar opciones válidas)
                sub1 = df_front[df_front["expiration"] == exp1].copy()
                sub2 = df_back[df_back["expiration"] == exp2].copy()

                # Obtener strikes comunes
                strikes1 = sub1["strike"].unique()
                strikes2 = sub2["strike"].unique()
                common_strikes = sorted(set(strikes1) & set(strikes2))

                if not common_strikes:
                    total_exp_pairs_skipped_no_common_strikes += 1
                    continue

                # Para cada strike común, crear un calendar
                for strike in common_strikes:
                    total_combos += 1

                    # Obtener filas para este strike
                    row1_df = sub1[np.isclose(sub1["strike"], strike, atol=1e-6)]
                    row2_df = sub2[np.isclose(sub2["strike"], strike, atol=1e-6)]

                    if row1_df.empty or row2_df.empty:
                        continue

                    row1 = row1_df.iloc[0]
                    row2 = row2_df.iloc[0]

                    # Obtener IVs
                    iv1 = get_iv_from_row(row1)
                    iv2 = get_iv_from_row(row2)

                    if iv1 is None or iv2 is None or iv1 <= 0 or iv2 <= 0:
                        continue

                    # Obtener deltas
                    delta_k1 = get_delta_from_row(row1)
                    delta_k2 = get_delta_from_row(row2)

                    # FILTRO 3: DELTA_K1 (valor absoluto para CALLs y PUTs)
                    if ENABLE_DELTA_K1_FILTER and delta_k1 is not None:
                        delta_k1_abs = abs(delta_k1)
                        if DELTA_K1_MIN is not None and delta_k1_abs < DELTA_K1_MIN:
                            continue
                        if DELTA_K1_MAX is not None and delta_k1_abs > DELTA_K1_MAX:
                            continue

                    # FILTRO 4: SPREAD (spread bid-ask relativo)
                    # Validar spread para ambas piernas del calendar
                    if ENABLE_SPREAD_FILTER:
                        if not leg_quote_ok(row1, MAX_SPREAD_REL):
                            continue
                        if not leg_quote_ok(row2, MAX_SPREAD_REL):
                            continue

                    # Calcular spread relativo para ambas piernas (para columna SPREAD)
                    spread_rel_k1, _, _, _ = get_spread_rel_from_row(row1)
                    spread_rel_k2, _, _, _ = get_spread_rel_from_row(row2)

                    # SPREAD del calendar: usar el máximo de ambas piernas (peor caso)
                    if spread_rel_k1 is not None and spread_rel_k2 is not None:
                        spread_calendar = max(spread_rel_k1, spread_rel_k2)
                    elif spread_rel_k1 is not None:
                        spread_calendar = spread_rel_k1
                    elif spread_rel_k2 is not None:
                        spread_calendar = spread_rel_k2
                    else:
                        spread_calendar = None

                    # Calcular Forward Factor
                    ff = compute_forward_factor(iv1, iv2, t1_years, t2_years)
                    ff_x1000 = int(round(ff * 1000)) if (np.isfinite(ff) and ff is not None) else None

                    # Obtener root symbols del CSV (fallback: root_for_exp)
                    if "root" in row1.index and pd.notna(row1["root"]):
                        root1_actual = str(row1["root"]).strip().upper()
                    else:
                        root1_actual = root_for_exp(exp1)

                    if "root" in row2.index and pd.notna(row2["root"]):
                        root2_actual = str(row2["root"]).strip().upper()
                    else:
                        root2_actual = root_for_exp(exp2)

                    # Generar URL de Optionstrat usando los roots reales
                    url_optionstrat = make_url_optionstrat_with_roots(
                        current_right, exp1, strike, exp2, strike, root1_actual, root2_actual
                    )

                    # Guardar resultado
                    results.append({
                        'date_us': date_us_str,
                        'date_es': date_es_str,
                        'file': filename,
                        'right': current_right,
                        'DTE1': dte1,
                        'DTE2': dte2,
                        'DTE_DIFF': dte2 - dte1,
                        'DTE_DIFF_PCT': round(dte_diff_pct, 6) if np.isfinite(dte_diff_pct) else None,
                        'SPREAD': round(spread_calendar, 6) if (spread_calendar is not None and np.isfinite(spread_calendar)) else None,
                        'expiration1': exp1,
                        'expiration2': exp2,
                        'strike': round(strike, 2),
                        'SPOT': round(spot, 2),
                        'DELTA_K1': round(delta_k1, 6) if (delta_k1 is not None and np.isfinite(delta_k1)) else None,
                        'DELTA_K2': round(delta_k2, 6) if (delta_k2 is not None and np.isfinite(delta_k2)) else None,
                        'IV1': round(iv1, 6) if np.isfinite(iv1) else None,
                        'IV2': round(iv2, 6) if np.isfinite(iv2) else None,
                        'T1_years': round(t1_years, 6),
                        'T2_years': round(t2_years, 6),
                        'Forward_Factor': round(ff, 6) if np.isfinite(ff) else None,
                        'Forward_Factor_x1000': ff_x1000,
                        'url': url_optionstrat
                    })

        print(f"    [OK] Combinaciones exploradas ({current_right}): {total_combos}")
        print(f"    [INFO] Pares de expiraciones procesados: {total_exp_pairs_processed}")
        print(f"    [INFO] Pares descartados por falta de strikes comunes: {total_exp_pairs_skipped_no_common_strikes}")

    print(f"  [OK] Calendarios válidos totales: {len(results)}")

    return results

def main():
    print("\n" + "="*70)
    print(">> V1 VIX CALENDAR FORWARD FACTOR SCANNER")
    print("="*70)
    print(f"\nCONFIGURACIÓN:")
    print(f"  - DTE front: {RANGE_A[0]}-{RANGE_A[1]}")
    print(f"  - DTE back: {RANGE_B[0]}-{RANGE_B[1]}")

    # Estado de filtros
    print(f"\n  FILTROS DE TRADING:")

    if ENABLE_DELTA_K1_FILTER:
        min_str = f"{DELTA_K1_MIN:.4f}" if DELTA_K1_MIN is not None else "-∞"
        max_str = f"{DELTA_K1_MAX:.4f}" if DELTA_K1_MAX is not None else "+∞"
        print(f"  - |DELTA_K1|: ACTIVADO [{min_str}, {max_str}]")
    else:
        print(f"  - |DELTA_K1|: DESACTIVADO")

    if ENABLE_DTE_DIFF_PCT_FILTER:
        print(f"  - DTE_DIFF_PCT: ACTIVADO [max <= {MAX_DTE_DIFF_PCT:.2%}]")
    else:
        print(f"  - DTE_DIFF_PCT: DESACTIVADO")

    if ENABLE_DTE_DIFF_FILTER:
        max_str = f"{MAX_DTE_DIFF}" if MAX_DTE_DIFF is not None else "+∞"
        print(f"  - DTE_DIFF: ACTIVADO [max <= {max_str} días]")
    else:
        print(f"  - DTE_DIFF: DESACTIVADO")

    if ENABLE_MAX_DTE1_FILTER:
        max_str = f"{MAX_DTE1}" if MAX_DTE1 is not None else "+∞"
        print(f"  - MAX_DTE1: ACTIVADO [max <= {max_str} días]")
    else:
        print(f"  - MAX_DTE1: DESACTIVADO")

    if ENABLE_SPREAD_FILTER:
        print(f"  - SPREAD: ACTIVADO [max <= {MAX_SPREAD_REL:.0%}]")
    else:
        print(f"  - SPREAD: DESACTIVADO")

    print(f"  - Tipo: CALL y PUT (ambos)")
    if IGNORE_TARGET_MINUTE:
        print(f"  - Snapshot: IGNORE_TARGET_MINUTE = True -> Procesando TODO el dia")
    else:
        print(f"  - Snapshot: {TARGET_HHMMS_US} (tolerancia {NEAREST_MINUTE_TOLERANCE} min)")

    # Determinar modo
    if FILENAME_TO_PROCESS is not None:
        print(f"  - Modo: Archivo específico")
        print(f"  - Patrón: {FILENAME_TO_PROCESS}")

        import glob
        search_pattern = os.path.join(DATA_DIR, FILENAME_TO_PROCESS)
        matched_files = glob.glob(search_pattern)

        if not matched_files:
            print(f"\n[X] No se encontraron archivos que coincidan con: {FILENAME_TO_PROCESS}")
            print(f"[X] Ruta de búsqueda: {DATA_DIR}")
            print(f"[X] Por favor verifica el nombre/patrón del archivo en la configuración")
            return

        chosen_files = [os.path.basename(f) for f in matched_files]
        chosen_files.sort()

        print(f"  - Archivos encontrados: {len(chosen_files)}")
        if len(chosen_files) <= 5:
            for f in chosen_files:
                print(f"    - {f}")
        else:
            for f in chosen_files[:3]:
                print(f"    - {f}")
            print(f"    ... y {len(chosen_files)-3} más")
    else:
        print(f"  - Modo: Archivos aleatorios")
        print(f"  - Archivos a procesar: {NUM_RANDOM_FILES}")

        # Listar archivos
        files_sorted = list_local_files(DATA_DIR)
        if not files_sorted:
            print(f"\n[X] No se encontraron archivos 30MINDATA_*.csv en: {DATA_DIR}")
            return

        print(f"\n[INFO] Archivos disponibles: {len(files_sorted)}")

        # Selección aleatoria
        k = min(NUM_RANDOM_FILES, len(files_sorted))
        chosen_files = random.sample(files_sorted, k=k)

        print(f"[INFO] Archivos seleccionados: {k}")

    # Procesar archivos
    all_results = []
    k = len(chosen_files)

    for idx, filename in enumerate(chosen_files, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{k}] Procesando: {filename}")
        print(f"{'='*70}")

        file_results = process_one_30min_file(filename, DATA_DIR)
        all_results.extend(file_results)

    # Consolidar resultados
    if not all_results:
        print("\n[X] No se generaron calendarios válidos")
        return

    print(f"\n{'='*70}")
    print(f"[CONSOLIDACIÓN]")
    print(f"{'='*70}")
    print(f"  Total calendarios generados: {len(all_results)}")

    # Crear DataFrame
    df_output = pd.DataFrame(all_results)

    # Ordenar por fecha y Forward_Factor_x1000 descendente
    # Convertir Forward_Factor_x1000 a numérico para asegurar ordenamiento correcto
    df_output['Forward_Factor_x1000'] = pd.to_numeric(df_output['Forward_Factor_x1000'], errors='coerce')

    df_output = df_output.sort_values(
        by=['date_us', 'Forward_Factor_x1000'],
        ascending=[True, False],
        na_position='last'  # Poner valores None/NaN al final
    ).reset_index(drop=True)

    # Reordenar columnas según especificación
    if not df_output.empty:
        # Definir orden preferido
        preferred_order = [
            'url',
            'DELTA_K1',
            'DELTA_K2',
            'Forward_Factor',
            'Forward_Factor_x1000',
            'right',
            'DTE1',
            'DTE2',
            'DTE_DIFF',
            'DTE_DIFF_PCT',
            'SPREAD',
            'expiration1',
            'expiration2',
            'strike',
            'SPOT',
            'IV1',
            'IV2',
            'T1_years',
            'T2_years',
            'date_us',
            'date_es',
            'file'
        ]

        # Construir lista final: columnas en orden preferido + resto + columnas al final
        cols_ordered = [c for c in preferred_order if c in df_output.columns]
        cols_remaining = [c for c in df_output.columns if c not in preferred_order]

        df_output = df_output[cols_ordered + cols_remaining]

    # Estadísticas
    print(f"\n{'='*70}")
    print(f"[ESTADÍSTICAS]")
    print(f"{'='*70}")

    ff_valid = df_output['Forward_Factor'].dropna()
    if len(ff_valid) > 0:
        print(f"  Forward Factor:")
        print(f"    - Min: {ff_valid.min():.6f}")
        print(f"    - Max: {ff_valid.max():.6f}")
        print(f"    - Media: {ff_valid.mean():.6f}")
        print(f"    - Mediana: {ff_valid.median():.6f}")
        print(f"    - Positivos (backwardation): {(ff_valid > 0).sum()} ({(ff_valid > 0).sum()/len(ff_valid)*100:.1f}%)")
        print(f"    - Negativos (contango): {(ff_valid < 0).sum()} ({(ff_valid < 0).sum()/len(ff_valid)*100:.1f}%)")

    # Guardar CSV
    ts_batch = datetime.now(TZ_ES).strftime("%Y%m%d_%H%M%S")
    output_name = f"VIX_Calendar_ForwardFactor_Scan_{ts_batch}.csv"
    output_path = DESKTOP / safe_filename(output_name)

    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*70}")
    print(f"[GUARDADO]")
    print(f"{'='*70}")
    print(f"  Archivo: {output_path}")
    print(f"  Filas: {len(df_output)}")

    # ================== GENERAR CSV "NO LOSE" ==================
    print(f"\n{'='*70}")
    print(f"[GENERANDO CSV NO LOSE]")
    print(f"{'='*70}")
    print(f"  Criterio: Para un mismo par de DTEs (DTE1, DTE2):")
    print(f"    - Debe haber mínimo 2 estructuras (2 strikes diferentes)")
    print(f"    - La diferencia |Forward_Factor| entre estructuras debe ser >= 0.2")

    # Filtrar solo filas con Forward_Factor válido
    df_no_lose = df_output[df_output['Forward_Factor'].notna()].copy()

    if df_no_lose.empty:
        print(f"\n  [X] No hay estructuras con Forward_Factor válido para filtrar")
    else:
        print(f"\n  [INFO] Estructuras con Forward_Factor válido: {len(df_no_lose)}")

        # Agrupar por DTE1, DTE2
        grouped = df_no_lose.groupby(['DTE1', 'DTE2'])

        # Filtrar grupos que cumplan el criterio
        valid_groups = []
        total_groups = 0
        groups_with_min_2 = 0
        groups_with_ff_diff = 0

        for (dte1, dte2), group in grouped:
            total_groups += 1

            # Debe haber al menos 2 estructuras (2 strikes diferentes)
            if len(group) < 2:
                continue

            groups_with_min_2 += 1

            # Calcular diferencia máxima de Forward_Factor en valor absoluto
            ff_values = group['Forward_Factor'].to_numpy()
            ff_min = np.min(ff_values)
            ff_max = np.max(ff_values)
            ff_diff = abs(ff_max - ff_min)

            # Verificar si la diferencia es >= 0.2
            if ff_diff >= 0.2:
                groups_with_ff_diff += 1
                # Agregar columna con la diferencia FF del grupo
                group_copy = group.copy()
                group_copy['FF_DIFF_GROUP'] = round(ff_diff, 6)
                valid_groups.append(group_copy)

        print(f"\n  [INFO] Grupos totales (DTE1 + DTE2): {total_groups}")
        print(f"  [INFO] Grupos con >= 2 estructuras: {groups_with_min_2}")
        print(f"  [INFO] Grupos con diferencia |FF| >= 0.2: {groups_with_ff_diff}")

        if valid_groups:
            df_no_lose_output = pd.concat(valid_groups, ignore_index=True)

            # Ordenar por FF_DIFF_GROUP descendente
            df_no_lose_output = df_no_lose_output.sort_values(
                by=['FF_DIFF_GROUP', 'DTE1', 'DTE2', 'Forward_Factor_x1000'],
                ascending=[False, True, True, False],
                na_position='last'
            ).reset_index(drop=True)

            # Reordenar columnas (FF_DIFF_GROUP al principio)
            if 'FF_DIFF_GROUP' in df_no_lose_output.columns:
                preferred_order_no_lose = [
                    'FF_DIFF_GROUP',
                    'url',
                    'DTE1',
                    'DTE2',
                    'DELTA_K1',
                    'DELTA_K2',
                    'Forward_Factor',
                    'Forward_Factor_x1000',
                    'right',
                    'DTE_DIFF',
                    'DTE_DIFF_PCT',
                    'SPREAD',
                    'expiration1',
                    'expiration2',
                    'strike',
                    'SPOT',
                    'IV1',
                    'IV2',
                    'T1_years',
                    'T2_years',
                    'date_us',
                    'date_es',
                    'file'
                ]

                cols_ordered_no_lose = [c for c in preferred_order_no_lose if c in df_no_lose_output.columns]
                cols_remaining_no_lose = [c for c in df_no_lose_output.columns if c not in preferred_order_no_lose]
                df_no_lose_output = df_no_lose_output[cols_ordered_no_lose + cols_remaining_no_lose]

            # Guardar CSV "No lose"
            output_name_no_lose = f"VIX_Calendar_ForwardFactor_Scan_{ts_batch}_NO_LOSE.csv"
            output_path_no_lose = DESKTOP / safe_filename(output_name_no_lose)

            df_no_lose_output.to_csv(output_path_no_lose, index=False, encoding='utf-8-sig')

            print(f"\n  [GUARDADO CSV NO LOSE]")
            print(f"    Archivo: {output_path_no_lose}")
            print(f"    Filas: {len(df_no_lose_output)}")
            print(f"    Columnas: {len(df_no_lose_output.columns)}")

            # Estadísticas del CSV NO LOSE
            print(f"\n  [ESTADÍSTICAS NO LOSE]")

            # Forward Factor stats
            ff_no_lose = df_no_lose_output['Forward_Factor'].dropna()
            if len(ff_no_lose) > 0:
                print(f"    Forward Factor (NO LOSE):")
                print(f"      - Min: {ff_no_lose.min():.6f}")
                print(f"      - Max: {ff_no_lose.max():.6f}")
                print(f"      - Media: {ff_no_lose.mean():.6f}")
                print(f"      - Mediana: {ff_no_lose.median():.6f}")
                print(f"      - Positivos: {(ff_no_lose > 0).sum()} ({(ff_no_lose > 0).sum()/len(ff_no_lose)*100:.1f}%)")
                print(f"      - Negativos: {(ff_no_lose < 0).sum()} ({(ff_no_lose < 0).sum()/len(ff_no_lose)*100:.1f}%)")

        else:
            print(f"\n  [X] No se encontraron grupos que cumplan el criterio NO LOSE")

    print(f"\n{'='*70}")
    print(f">> PROCESO COMPLETADO")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
