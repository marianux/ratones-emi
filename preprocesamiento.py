#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:48:58 2025

@author: mariano
"""

#%% Implementación en Python simil RT

import scipy.signal as sig 


def promediador_rt_init( xx, DD, UU ):

    # ventana de selección de UU muestras por el sobremuestreo
    hh_u = np.zeros(DD * UU)
    hh_u[::UU] = 1
    hh_u = np.flip(hh_u)
    hh_u = np.tile( hh_u[:, np.newaxis], xx.shape[1])

    
    # se asume como salida el mismo valor medio para las primeras UU muestras
    yy_ci = np.zeros((UU, xx.shape[1]))
    yy_ci[:,:] = np.sum( xx[:(DD * UU), :] * hh_u, axis = 0)

    # se consideran las primeras DD muestras a una frec de muestreo UU veces más
    # elevada.
    xx_ci = xx[:(DD * UU),:]

    return( (xx_ci, yy_ci) )

def promediador_rt( xx, DD, UU, xx_ci, yy_ci, kk_offset = 0):
    
    NN = xx.shape[0]

    # resultaron ser importante las condiciones iniciales
    yy = np.zeros_like(xx)
    # yy = np.ones_like(xx) * xx[0] * DD * UU

    if(kk_offset == 0):

        # condiciones iniciales
        for kk in range(UU):
    
            # Calcula la salida según la ecuación recursiva
            yy[kk,:] = xx[kk,:] \
                      - xx_ci[kk,:] \
                      + yy_ci[kk,:]
              
        # extiendo las salidas al mismo valor que yy[UU]

        yy[kk:DD * UU,:] = yy[kk,:]
        
        # vector para filtrar muestras
        hh_u = np.zeros(DD * UU)
        hh_u[::UU] = 1
        hh_u = np.flip(hh_u)
        hh_u = np.tile( hh_u[:, np.newaxis], xx.shape[1])

        # inicio de la recursión
        for kk in range(DD * UU, (DD * UU) + UU ):
    
            ii = kk-1
            # Calcula la salida según la ecuación recursiva
            yy[ii,:] = np.sum(xx[kk-(DD * UU):kk,:] * hh_u, axis = 0)

    else:
        # para todos los bloques restantes salvo el primero
           
        for kk in range(UU):
    
            # Calcula la salida según la ecuación recursiva
            yy[kk,:] = xx[kk,:] \
                      - xx_ci[kk,:] \
                      + yy_ci[kk,:]
        
        for kk in range(UU, DD * UU):

            # Calcula la salida según la ecuación recursiva
            yy[kk,:] = xx[kk,:] \
                      - xx_ci[kk,:] \
                      + yy[(kk - UU),:]
    
        #
        kk += 1
    
    
    # for kk in range(NN):
    for kk in range(kk, NN):

        # Calcula la salida según la ecuación recursiva
        yy[kk,:] = xx[kk,:]  \
                  - xx[kk - DD * UU,:] \
                  + yy[kk - UU,:]
    
    # calculo las condiciones iniciales del siguiente bloque
    xx_ci = xx[(NN - DD * UU):,:]
    yy_ci = yy[(NN - UU):,:]

    # escalo y devuelvo
    return( (yy.copy()/DD, xx_ci.copy(), yy_ci.copy()) )

def filtro_peine_DCyArmonicas( xx, DD = 16, UU = 2, MA_stages = 2 ):

    NN = xx.shape[0]

    ###############################################################################
    # estimación del valor medio en ventanas de DD muestras y sobremuestreo por UU

    # Se plantea la posibilidad de una implementación en un entorno de memoria
    # limitada que obligue al procesamiento de "bloque a bloque"
    
    # se calculan condiciones iniciales para el primer bloque moving averager (MA)
    # en total habrá MA_stages en cascada.
    xx_ci, yy_ci = promediador_rt_init( xx, DD, UU )

    yy = np.zeros_like(xx)
    
    # se procesa cada bloque por separado y se concatena la salida
    for jj in range(0, NN, block_s):
    
        yy_aux, xx_ci, yy_ci = promediador_rt( xx[jj:jj+block_s], DD, UU, xx_ci, yy_ci, kk_offset=jj)

        yy[jj:jj+block_s,:] = yy_aux

    # cascadeamos MA_stages-1 más
    for ii in range(1, MA_stages):

        # se calculan condiciones iniciales
        xx_ci, yy_ci = promediador_rt_init( yy, DD, UU )
        
        for jj in range(0, NN, block_s):
        
            yy_aux, xx_ci, yy_ci = promediador_rt( yy[jj:jj+block_s], DD, UU, xx_ci, yy_ci, kk_offset=jj)
        
            yy[jj:jj+block_s, :] = yy_aux

    #############################################################
    # demora de la señal xx y resta de la salida del último MA
    
    xx_aux = np.roll(xx, int((DD-1)/2*MA_stages*UU), axis=0 )
    yy = xx_aux - yy
    return( yy )

def blackman_tukey(x,  M = None):    
    
    N = x.shape[0]
    
    if M is None:
        M = N//5
    
    r_len = 2*M-1

    Px = np.zeros_like(x)
    
    for ii in range(x.shape[1]):    
        
        # hay que aplanar los arrays por np.correlate.
        # usaremos el modo same que simplifica el tratamiento
        # de la autocorr
        xx_rav = x[:,ii].ravel()[:r_len]
    
        r = np.correlate(xx_rav, xx_rav, mode='same') / r_len
    
        Px[:,ii] = np.abs(np.fft.fft(r * sig.windows.blackman(r_len), n = N) )

    return Px


#%% Aplicación a señales ratoncitos de emi

import numpy as np
import matplotlib.pyplot as plt
import os
from aip_detector import aip_detector

import neurokit2 as nk

# configuración del filtro de Lyons
dd = 64
uu = 10
ma_st = 2

# demora teórica del filtro de Rick Lyons.
demora_rl = int((dd-1)/2*ma_st*uu)

# Voy a resamplear para que el cero del filtro de Lyons coincida con
# los 50 Hz.
# upsampling
uu_signal = 2
fs = 250 # Hz (250 x uu_signal)


carpeta_ecg = '/home/mariano/Descargas/ratones emi/Datos Incógnita/solo_ecg'
# Obtener todos los archivos con extensión .ecg en el directorio actual
archivos_ecg = [f for f in os.listdir(carpeta_ecg) if f.endswith('_filt_qrs.npz')]

# Procesar cada archivo con mi_funcion
for archivo in archivos_ecg:
    
    archivo = archivo[:-13]
    filepath = os.path.join(carpeta_ecg, archivo)

    print(f'Procesando {archivo}')
    
#%% conversión de formato y filtrado
    
    archivo_filt = os.path.splitext(archivo)[0] + "_filt"
    filepath_filt = os.path.join(carpeta_ecg, archivo_filt)

    archivo_filt_qrs = archivo_filt + "_qrs"
    filepath_filt_qrs = os.path.join(carpeta_ecg, archivo_filt_qrs)

    archivo_packs = archivo_filt_qrs + "_packs"
    filepath_packs = os.path.join(carpeta_ecg, archivo_packs)

    bYaProcesado = np.any( [os.path.exists(filepath_filt + '.npz'), 
                            os.path.exists(filepath_filt_qrs + '.npz'), 
                            os.path.exists(filepath_packs + '.npz'), 
                            ])
    
    if not bYaProcesado:
        
        # Leer los datos binarios y reorganizar para 12 derivaciones
        with open(filepath, 'rb') as f:
            ecg_data = np.fromfile(f, dtype=np.int16, offset=4096).astype(np.float64)
    
        
        ecg_chan = 13
        
        num_samples = len(ecg_data) // ecg_chan
        ecg_channels = ecg_data.reshape((-1, ecg_chan))
        ecg_channels = ecg_channels[:,0:3]
    
        if uu_signal > 1:
            ecg_channels = sig.resample_poly(ecg_channels, up=uu_signal, down=1, axis=0)
        
        cant_muestras = ecg_channels.shape[0]
        
        block_s = cant_muestras
        
        ecg_channels_filtrado = filtro_peine_DCyArmonicas( ecg_channels, DD = dd, UU = uu, MA_stages = ma_st )

        if uu_signal > 1:
            ecg_channels_filtrado = sig.resample_poly(ecg_channels_filtrado, up=1, down=uu_signal, axis=0)
        
        # Guardar la matriz en formato binario de numpy
        np.savez(filepath_filt, ECG = ecg_channels_filtrado)
    
    
    
    
    # Visualización debug.
    #
    # plt.close('all')
    
    # regs_interes = ( 
    #        [5000, 7000], # muestras
    #        [20000, 22000], # muestras
    #         )
    
    # # for ii in regs_interes:
        
    # #     # intervalo limitado de 0 a cant_muestras
    # #     zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
        
    # #     plt.figure()
    # #     plt.plot(zoom_region, ecg_channels[zoom_region,:], ':', label='ECG', alpha=0.5)
    
    # #     # FIR con corrección de demora
    # #     plt.plot(zoom_region, ecg_channels_filtrado[np.clip(zoom_region+demora_rl,0,cant_muestras-1)], label='final')
        
    # #     plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    # #     plt.ylabel('Adimensional')
    # #     plt.xlabel('Muestras (#)')
        
    # #     axes_hdl = plt.gca()
    # #     axes_hdl.legend()
                
    # #     plt.show()

    # zoom_region = np.arange(0, cant_muestras, dtype='uint')

    # plt.figure(1)
    # plt.clf()
    # plt.plot( ecg_channels, ':', label='ECG', alpha=0.5)
    # plt.plot(zoom_region, ecg_channels_filtrado[np.clip(zoom_region+demora_rl,0,cant_muestras-1), :], label='filtrado')
    # axes_hdl = plt.gca()
    # axes_hdl.legend()
    # plt.show()

    
    # # # psd_xx = blackman_tukey( ecg_channels,          num_samples*2//5 )
    # # # psd_yy = blackman_tukey( ecg_channels_filtrado, num_samples*2//5  )

    # # psd_xx = 1/num_samples*np.abs(np.fft.fft(ecg_channels, axis=0))
    # # psd_yy = 1/num_samples*np.abs(np.fft.fft(ecg_channels_filtrado, axis=0))

    # # ff = np.arange(start=0, stop=fs/2, step = fs/num_samples/uu_signal)

    # # psd_xx = psd_xx[:ff.shape[0]]
    # # psd_yy = psd_yy[:ff.shape[0]]

    # # plt.figure(2)
    # # plt.clf()

    # # plt.plot(ff, 20*np.log10(psd_yy/psd_xx), ':^', label= 'filtro', alpha=0.5)

    # # plt.plot(ff, 20*np.log10(psd_xx), ':o', label= 'ECG', alpha=0.5)

    # # plt.legend()
    # # # plt.axis([-10, 510, -100, 0 ])
    # # # plt.show()



    # # plt.figure(3)
    # # plt.clf()
    # # plt.plot( ecg_channels, ':', label='ECG', alpha=0.5)
    # # plt.plot(zoom_region, ecg_channels_filtrado[np.clip(zoom_region+demora_rl,0,cant_muestras-1), :], label='filtrado')
    # # axes_hdl = plt.gca()
    # # axes_hdl.legend()
    # # plt.show()
    
    # pass

#%% detección de QRS

    archivo_filt_qrs = archivo_filt + "_qrs"

    filepath_filt_qrs = os.path.join(carpeta_ecg, archivo_filt_qrs)

    bYaProcesado = np.any( [ os.path.exists(filepath_filt_qrs + '.npz'), 
                             os.path.exists(filepath_packs + '.npz'), 
                            ])
    
    if not bYaProcesado:
    
        npz_file = np.load(filepath_filt + '.npz')
        
        ecg_channels_filtrado = npz_file['ECG']
        
        ECG_header = { 'freq' : fs,
                      'nsamp' : ecg_channels_filtrado.shape[0],
                      'nsig' : ecg_channels_filtrado.shape[1],
                      'recname' : os.path.splitext(archivo)[0],
                      'desc' : ['I', 'II', 'III']
                      }
        
        # config para ratones provisoria
        my_pattern = 0.
        
        params_in = {
            "arb_pattern" : my_pattern,  # Patrón arbitrario a buscar
            "trgt_width" : 0.04,  # Ancho de QRS en segundos
            "lp_size" : np.round(0.015 * 250),  # Ancho del filtro pasabajos luego del filtro adaptado. Default: 0. ó 1.2*trgt_width
            "trgt_min_pattern_separation" : 0.06,  # Separación mínima entre patrones en segundos
            "trgt_max_pattern_separation" : 0.2,  # Separación máxima entre patrones en segundos
            "stable_RR_time_win" : 2,  # Ventana de tiempo para ritmo estable en segundos
            "final_build_time_win" : 20,  # Ventana de tiempo para construir detección final en segundos
            "powerline_interference" : np.nan,  # Interferencia de línea de potencia (Hz)
            "max_patterns_found" : 3,  # Número máximo de patrones detectados
            "sig_idx" : np.arange(ECG_header['nsig'])  # Índices de señales
        }
            
        QRS_detections = aip_detector(ecg_channels_filtrado, ECG_header, payload_in = params_in)
    
        # plt.figure(1)
        # plt.plot(ecg_channels_filtrado)
        # aux_qrs = QRS_detections['aip_guess_I']['time']
        # plt.plot(aux_qrs, ecg_channels_filtrado[aux_qrs, 0], 'bd', label="QRS I")
        # aux_qrs = QRS_detections['aip_guess_II']['time']
        # plt.plot(aux_qrs, ecg_channels_filtrado[aux_qrs, 1], 'gd', label="QRS II")
        # aux_qrs = QRS_detections['aip_guess_III']['time']
        # plt.plot(aux_qrs, ecg_channels_filtrado[aux_qrs, 2], 'rd', label="QRS III")
        # plt.legend()
        # plt.grid()
        # plt.show()
    
        # Guardar la matriz en formato binario de numpy
        np.savez(filepath_filt_qrs, ECG = ecg_channels_filtrado, QRS_det = QRS_detections, allow_pickle=True)

#%% QRS promedios


    archivo_packs = archivo_filt_qrs + "_packs"

    filepath_packs = os.path.join(carpeta_ecg, archivo_packs)

    bYaProcesado = np.any( [ os.path.exists(filepath_packs + '.npz'), 
                            ])
    
    if not bYaProcesado:
    
        npz_file = np.load(filepath_filt_qrs + '.npz', allow_pickle=True)

        ecg_clean = npz_file['ECG']

        QRS_det = npz_file['QRS_det'].item()
        
        QRS_pack = {'AnnNames': QRS_det['AnnNames']}
        QRS_median = {'AnnNames': QRS_det['AnnNames']}
        
        time_ref =  np.arange(-0.05, stop=0.08, step = 1/250)
        
        for ii, this_lead in enumerate(QRS_det['AnnNames']):
            
            rpeaks = QRS_det[this_lead]['time']
    
            epochs = nk.epochs_create(ecg_clean[:,ii], events=rpeaks, epochs_start=-0.05, epochs_end=0.08, sampling_rate=250)

            this_QRS_pack = nk.epochs_to_array(epochs)
            
            this_QRS_pack = this_QRS_pack - np.median(this_QRS_pack,axis = 0)
            
            QRS_pack[this_lead] = this_QRS_pack
            
            this_heartbeat_median = np.median(this_QRS_pack,axis = 1)
            
            QRS_median[this_lead] = this_heartbeat_median
            
            
            # plt.close(ii)
            # plt.figure(ii)
            # plt.clf()
            # plt.plot(time_ref, this_QRS_pack, color="gray", linewidth=0.5, alpha=0.5)
            
            # # Graficar la señal promedio en rojo grueso
            # plt.plot(time_ref, this_heartbeat_median, color="red", linewidth=2, label="Heartbeat Median")
            
            # plt.legend()
            # plt.title(this_lead)
            # plt.show()
            

        # Guardar la matriz en formato binario de numpy
        np.savez(filepath_packs, 
                 ECG = ecg_clean, 
                 QRS_det = QRS_det, 
                 QRS_median = QRS_median, 
                 QRS_pack = QRS_pack, 
                 allow_pickle=True)

        pass

#%%

    npz_file = np.load(filepath_packs + '.npz', allow_pickle=True)

    ecg_clean = npz_file['ECG']

    QRS_det = npz_file['QRS_det'].item()
    
    QRS_median = npz_file['QRS_median'].item()
    
    QRS_pack = npz_file['QRS_pack'].item()
    
    QRS_det = npz_file['QRS_det'].item()

    time_ref =  np.arange(-0.05, stop=0.08, step = 1/250)
    
    for ii, this_lead in enumerate(QRS_det['AnnNames']):
        
        this_QRS_pack = QRS_pack[this_lead]
        
        this_heartbeat_median = QRS_median[this_lead]
        
        plt.figure(ii)
        plt.clf()
        plt.plot(time_ref, this_QRS_pack, color="gray", linewidth=0.5, alpha=0.5)
        
        # Graficar la señal promedio en rojo grueso
        plt.plot(time_ref, this_heartbeat_median, color="red", linewidth=2, label="Heartbeat Median")
        
        plt.ylim((-50, 60))
        plt.legend()
        plt.title(this_lead)

    plt.show()
    pass
