#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:04:52 2025

@author: mariano
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, lfilter, medfilt
from scipy.signal.windows import gaussian

def thr_calc(rise_detector, trgt_min_pattern_separation, freq):
    """
    Calcula el umbral basado en el percentil de los valores de rise_detector.
    """

    initial_thr = 30

    # Cálculo del umbral inicial basado en percentiles
    actual_thr = np.percentile(rise_detector, initial_thr)

    # Reemplazo de modmax con scipy.signal.find_peaks
    peaks, _ = find_peaks(rise_detector, height=actual_thr, distance=np.round(trgt_min_pattern_separation * freq))
    max_values = rise_detector[peaks]

    if len(max_values) == 0:
        # Posible grabación plana
        actual_thr = np.nan
    else:
        # Cálculo de percentiles y paso de la grilla
        prctile_grid = np.percentile(max_values, np.arange(1, 101))
        grid_step = np.median(np.diff(prctile_grid))

        if grid_step == 0:
            actual_thr = np.nan
        else:
            idx = np.argmin(np.abs(prctile_grid - (prctile_grid[0] + grid_step * 10)))
            actual_thr = prctile_grid[idx]

        # Construcción de la grilla de umbrales
        thr_grid = np.arange(actual_thr, max(max_values) + grid_step, grid_step)
        hist_max_values, _ = np.histogram(max_values, bins=thr_grid)
        first_bin_idx = 1

        # Reemplazo de modmax con scipy.signal.find_peaks
        thr_idx, _ = find_peaks(hist_max_values, height=0)
        thr_max = hist_max_values[thr_idx]

        if len(thr_max) == 0:
            actual_thr = np.nan
        else:
            # Centro de masa de la distribución, probablemente donde se encuentran los patrones buscados
            thr_idx_expected = np.floor(np.dot(thr_idx, thr_max) / np.sum(thr_max))

            aux_seq = np.arange(len(hist_max_values))

            # MPS0018 recording de la base de datos Basel VIII obligó a agregar esta condición
            # Probablemente un caso donde no se encontraron máximos claros sobre el umbral de ruido
            mask = (aux_seq >= first_bin_idx) & (aux_seq < thr_idx_expected)
            min_hist_max_values = np.min(hist_max_values[mask])

            # En caso de múltiples índices coincidentes, se elige el centro de los valores mínimos
            thr_min_idx_candidates = np.where(mask & (hist_max_values == min_hist_max_values))[0]

            if len(thr_min_idx_candidates) == 0:
                actual_thr = np.nan
            else:
                thr_min_idx = np.round(np.mean(thr_min_idx_candidates))
                actual_thr = thr_grid[int(thr_min_idx)]

    return actual_thr


def colvec(x):
    """
    Convierte un array en un vector columna.
    """
    return np.reshape(x, (-1, 1))



from scipy.interpolate import CubicSpline

def MedianFiltSequence(x, y, filter_win):
    """
    Aplica un filtro de mediana a una secuencia interpolada.

    Parámetros:
    - x: Puntos de muestreo originales.
    - y: Valores correspondientes a los puntos x.
    - filter_win: Tamaño de la ventana del filtro de mediana.

    Retorna:
    - y_filt: Secuencia filtrada.
    """

    # Sobremuestreo de la secuencia original
    oversampling_ratio = np.round(np.median(np.diff(x)) / 4)

    x_interp = np.arange(0, max(x), oversampling_ratio)
    spline_interpolator = CubicSpline(x, y)
    y_interp = spline_interpolator(x_interp)

    # Aplicación del filtro de mediana, aseguro kernel_size impar
    y_filt = medfilt(y_interp.flatten(), kernel_size=(round(filter_win / oversampling_ratio/2)*2)+1)

    # Mapeo de la señal filtrada a los índices originales
    aux_idx = np.floor(x / oversampling_ratio).astype(int)
    y_filt = y_filt[aux_idx]

    return y_filt



def RR_calculation(QRS_detections, sampling_rate, RRseq_filt_win = 2., RR_filt=None):
    """
    Calcula la secuencia de intervalos RR a partir de las detecciones de QRS.

    Parámetros:
    - QRS_detections: Array con los tiempos de detección de los complejos QRS.
    - sampling_rate: Frecuencia de muestreo de la señal.
    - RR_filt: Secuencia RR filtrada (opcional).

    Retorna:
    - RR: Secuencia de intervalos RR.
    - RR_filt: Secuencia RR filtrada.
    """

    # Cálculo de la serie RR
    RR = np.diff(QRS_detections)
    RR = np.insert(RR, 0, RR[0])  # Equivalente a [RR(1,:); RR] en MATLAB

    # Si no se proporciona RR_filt o su longitud no coincide con QRS_detections, recalcularlo
    if RR_filt is None or len(RR_filt) != len(QRS_detections):
        # Resampleo de la secuencia RR (cómputo pesado)
        RR_filt = MedianFiltSequence(QRS_detections, RR, RRseq_filt_win * sampling_rate)

    # Eliminar discontinuidades usando los próximos latidos
    gap_relative_time = 3  # Veces el intervalo RR medio
    aux_val = RR / RR_filt
    aux_idx = np.where(aux_val >= gap_relative_time)[0]

    for ii in aux_idx:
        # Buscar el primer latido después del actual con un valor dentro del umbral
        next_idx = np.where((QRS_detections > QRS_detections[ii]) & (aux_val < gap_relative_time))[0]
        aux_idx_match = next_idx[0] if len(next_idx) > 0 else None

        if aux_idx_match is None:
            # Si no se encuentra, buscar el último latido antes del actual dentro del umbral
            prev_idx = np.where((QRS_detections < QRS_detections[ii]) & (aux_val < gap_relative_time))[0]
            aux_idx_match = prev_idx[-1] if len(prev_idx) > 0 else None

        if aux_idx_match is not None:
            RR[ii] = RR[aux_idx_match]

    return RR, RR_filt


import re

def aip_detector(ECG_matrix, ECG_header, ECG_start_offset = 1, payload_in = None):
    """
    aip_detector: Detector de impulsos pseudoperiódicos arbitrarios.
    """

    if not isinstance(payload_in , (dict, type(None))):
        raise ValueError('payload_in debe ser un diccionario o None.')

    # Inicialización del payload
    payload = {
            "series_quality": [],
            "AnnNames": [],
            "ratios": [],
            "estimated_labs": []
    }

    # Configuración de parámetros por defecto
    default_params = {
        "arb_pattern" : 0.,  # Patrón arbitrario a buscar
        "trgt_width" : 0.06,  # Ancho de QRS en segundos
        "lp_size" : 0.,  # Ancho del filtro pasabajos luego del filtro adaptado. Default: 0. ó 1.2*trgt_width
        "trgt_min_pattern_separation" : 0.3,  # Separación mínima entre patrones en segundos
        "trgt_max_pattern_separation" : 2,  # Separación máxima entre patrones en segundos
        "stable_RR_time_win" : 2,  # Ventana de tiempo para ritmo estable en segundos
        "final_build_time_win" : 20,  # Ventana de tiempo para construir detección final en segundos
        "powerline_interference" : np.nan,  # Interferencia de línea de potencia (Hz)
        "max_patterns_found" : 3,  # Número máximo de patrones detectados
        "sig_idx" : list(range(ECG_header['nsig'] ))  # Índices de señales
    }

    # Asignar valores por defecto si no están en payload_in
    for key, value in default_params.items():
        if key not in payload_in:
            payload_in[key] = value

    ## Preprocesamiento 
    lead_names = ECG_header["desc"]

    # Desambiguación de nombres de derivaciones
    # lead_names = [re.sub(r'\W*(\w+)\W*', r'\1', desc) for desc in ECG_header["desc"]]
    # lead_names = [re.sub(r'\W', '_', name) for name in lead_names]

    # # Identificación de nombres únicos
    # unique_names, aux_idx = np.unique(lead_names, return_inverse=True)
    # aux_val = len(unique_names)

    # if aux_val != ECG_header["nsig"]:
    #     for ii in range(aux_val):
    #         bAux = aux_idx == ii
    #         aux_matches = np.sum(bAux)
    #         if aux_matches > 1:
    #             counter = np.arange(1, aux_matches + 1).astype(str)
    #             lead_names = np.array(lead_names, dtype=object)
    #             lead_names[bAux] = [name + 'v' + num for name, num in zip(lead_names[bAux], counter)]

    # # Aplicar limpieza final a los nombres
    # lead_names = [re.sub(r'\W*(\w+)\W*', r'\1', name) for name in lead_names]
    # lead_names = [re.sub(r'\W', '_', name) for name in lead_names]

    ## Inicio del procesamiento
    if payload_in["arb_pattern"] == 0:
        pattern_size = 2 * np.round(payload_in["trgt_width"] / 2 * ECG_header["freq"]) + 1  # Forzar número impar
        pattern_coeffs = np.diff(gaussian(pattern_size + 1, std=pattern_size / 6)) * gaussian(pattern_size, std=pattern_size / 6)
        first_pattern_coeffs = pattern_coeffs
    else:
        first_pattern_coeffs = payload_in["arb_pattern"]
    

    if payload_in["lp_size"] == 0:
        lp_size = np.round(1.2 * pattern_size)
    else:
        lp_size = payload_in["lp_size"]

    lp_size = lp_size.astype(int)
    
    for this_sig_idx in payload_in["sig_idx"]:

        
        ## Búsqueda inicial de patrones
        rise_detector = lfilter(first_pattern_coeffs, 1, np.flipud(ECG_matrix[:, this_sig_idx]))
        rise_detector = lfilter(first_pattern_coeffs, 1, np.flipud(rise_detector))
        rise_detector = lfilter(np.ones(lp_size) / lp_size, 1, np.flipud(abs(rise_detector)))
        rise_detector = lfilter(np.ones(lp_size) / lp_size, 1, np.flipud(rise_detector))

        # # Figura 4: Comparación del detector de picos con la señal ECG
        # plt.figure(4)
        # plt.plot(rise_detector / np.max(rise_detector), label="rise detector")
        # plt.plot(ECG_matrix[:, this_sig_idx] / np.max(ECG_matrix[:, this_sig_idx]), label="ECG")
        # plt.legend()
        # plt.title(f"Rise Detector vs ECG - Lead {lead_names[this_sig_idx]}")
        # plt.xlabel("Muestras")
        # plt.ylabel("Amplitud Normalizada")
        # plt.grid()

        # # Figura 3: Visualización de la señal ECG y el rise detector escalado
        # plt.figure(3)
        # plt.plot(ECG_matrix[:, 0])  # ECG canal 1
        # plt.xlim([0, len(ECG_matrix[:, 0])])
        # ylims = plt.ylim()
        # aux_sc = 0.8 * np.diff(ylims) / (np.max(rise_detector) - np.min(rise_detector))
        # aux_off = ylims[0] + 0.6 * np.diff(ylims)
        # plt.plot(rise_detector * aux_sc + aux_off, label="Rise Detector Escalado")
        # plt.ylim(ylims)
        # plt.legend()
        # plt.title("ECG y Rise Detector Escalado")
        # plt.xlabel("Muestras")
        # plt.ylabel("Amplitud")
        # plt.grid()

        # Cálculo del umbral para detección de patrones
        actual_thr = thr_calc(rise_detector, payload_in["trgt_min_pattern_separation"], ECG_header["freq"])

        if np.isnan(actual_thr):
            print(f"\nRecording {ECG_header['recname']}, lead {lead_names[this_sig_idx]}, no patterns above threshold found. (Current {payload_in['stable_RR_time_win']} seconds).\n")
            continue

        # Reemplazo de modmax con scipy.signal.find_peaks
        first_detection_idx, _ = find_peaks(rise_detector, height=actual_thr, distance=np.round(payload_in["trgt_min_pattern_separation"] * ECG_header["freq"]))

        ## Búsqueda de segmentos estables
        if len(first_detection_idx) <= 1:
            print(f"\nRecording {ECG_header['recname']}, lead {lead_names[this_sig_idx]}, Not enough patterns found over noise floor.\n")
            continue

        # # Cálculo de la serie RR
        # RRserie, _ = RR_calculation(first_detection_idx, ECG_header["freq"], RRseq_filt_win = payload_in["trgt_max_pattern_separation"])

        # # Aplicación de filtro de mediana
        # RRserie_filt = MedianFiltSequence(first_detection_idx, RRserie, np.round(2*payload_in["trgt_max_pattern_separation"] * ECG_header["freq"]) )

        # # Cálculo de dispersión de RR
        # RR_scatter = np.abs(RRserie - RRserie_filt) / RRserie_filt

        # # Encontrar las regiones más estables
        # RR_thr = np.percentile(RR_scatter, 50)

        # plt.figure()
        # plt.plot(ECG_matrix[:, this_sig_idx], label='lead '+lead_names[this_sig_idx])
        # plt.plot(first_detection_idx, ECG_matrix[first_detection_idx, this_sig_idx], 'rd', label="QRS detections")
        # plt.legend()
        # plt.grid()
        # plt.show()


        str_aux = f"aip_guess_{lead_names[this_sig_idx]}"

        payload['AnnNames'].append(str_aux)
        payload[str_aux] = {"time": first_detection_idx + ECG_start_offset - 1}

        # Cálculo del error cuadrático medio (MSE) entre RRserie y RRserie_filt
        # RRserie_mean_sq_error = np.mean((RRserie - RRserie_filt) ** 2)
        # payload["series_quality"]["ratios"].append(RRserie_mean_sq_error)
        # payload["series_quality"]["estimated_labs"].append([])

        # Normalización del ratio de calidad
        # max_ratio = max(payload["series_quality"]["ratios"])
        # payload["series_quality"]["ratios"] = [1 - (r / max_ratio) for r in payload["series_quality"]["ratios"]]

    return payload
