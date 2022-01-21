#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:32:12 2021

@author: alejomonbar
"""
import numpy as np


Re_lam_max = 1187 # The laminar region is defined by a Reynolds number lower than this
Re_tur_min = 3000 # Zirgang and Silvester give this relation.

def dpdz(f,G,ID,p_mix): #kPa/m
    return (2 * f * G**2) / (1000 * ID * p_mix)

def dpdz2(f,G,ID,p_L,phi2,x):#kPa/m Sun
    return 2 * f * ((G * (1 - x))**2/(1000 * ID * p_L)) * phi2

def f_turb(Re_mix,e,ID):
    am = (2.457 * np.log(1 / ((7 / Re_mix)**0.9 + (0.27 * e / ID))))**16
    bm = (37530/Re_mix)**16
    return 2*((8/Re_mix)**12 + 1/(am+bm)**(3/2.0))**(1/12.0)

def f_lam(Re_mix,e,ID):
    am = (2.457 * np.log(1 / ((7 / Re_mix)**0.9 + (0.27 * e / ID))))**16
    bm = (37.53/Re_mix)**16
    return 2*((8/Re_mix)**12 + 1/(am+bm)**(3/2.0))**(1/12.0)

def Re_phase(G,ID,mu_mix):
    return (G * ID) / mu_mix

def X(dpdz_L,dpdz_v):
    return np.sqrt(dpdz_L/dpdz_v)

def C(Re_L,Re_v,x): #mixt
    return 1.79 * (Re_v / Re_L)**0.4 * ((1.0 - x) / x)**0.5  
def phi2(C,X):
    return 1 + C / X**(1.19) + 1.0 / X**2

def dpdz_sun(x, ID, G, e, p_liq, p_vap, vis_liq, vis_vap):
   
    n = len(x)
    
    f_liq = np.zeros((n, 1))
    f_vap = np.zeros((n, 1))
    #The two-phases
    G_liq = G * (1 - x)
    G_vap = G * x
    Re_vap = Re_phase(G_vap, ID, vis_vap)
    Re_liq = Re_phase(G_liq, ID, vis_liq)
    # friction Laminar
    f_liq[Re_liq <= Re_lam_max] = f_lam(Re_liq[Re_liq <= Re_lam_max], e, ID)
    f_vap[Re_vap <= Re_lam_max] = f_lam(Re_vap[Re_vap <= Re_lam_max], e, ID)
    # friction Turbulent
    f_liq[Re_liq >= Re_tur_min] = f_turb(Re_liq[Re_liq >= Re_tur_min], e, ID)
    f_vap[Re_vap >= Re_tur_min] = f_turb(Re_vap[Re_vap >= Re_tur_min], e, ID)
    # friction transition
    ten_Re = np.polyfit([Re_lam_max, Re_tur_min],[f_lam(Re_lam_max, e, ID), f_turb(Re_tur_min, e, ID)], 1)
    pol_f = np.poly1d(ten_Re)
    arg1 = (Re_lam_max<Re_liq) * (Re_liq < Re_tur_min)
    f_liq[arg1] = pol_f(Re_liq[arg1])
    arg2 = (Re_lam_max < Re_vap) * (Re_vap < Re_tur_min)
    f_vap[arg2] = pol_f(Re_vap[arg2])
    
    dpdz_liq = dpdz(f_liq, G_liq, ID, p_liq)        
    #vapor phase
    
    
    dpdz_vap = dpdz(f_vap, G_vap, ID, p_vap)
    X_m = X(dpdz_liq,dpdz_vap)
    C_m = C(Re_liq,Re_vap,x)
    phi_L = phi2(C_m,X_m)
    dpdz_sun = {}
    dpdz_sun[1] = x
    dpdz_sun[2] = e * np.ones(x.shape)
    dpdz_sun[3] = G * np.ones(x.shape)
    dpdz_sun[5] = ID * np.ones(x.shape)
    dpdz_sun[6] = Re_liq
    dpdz_sun[8] = f_liq
    dpdz_sun[22] = dpdz2(f_liq,G,ID,p_liq,phi_L,x) #Sun
    
    return  dpdz_sun

