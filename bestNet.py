#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 06:52:51 2021

@author: alejomonbar
"""
import numpy as np

def mapminmax_apply(x, x_norm):
    xn = x - x_norm["xoffset"];
    xn *= x_norm["gain"];
    xn += x_norm["ymin"];
    return xn

def mapminmax_reverse(y, y_norm):
    yn = y - y_norm["ymin"];
    yn /= y_norm["gain"];
    yn += y_norm["xoffset"];
    return yn

def tansig(x):
    return 2 / (1 + np.exp(-2 * x)) - 1

def net(x, params):
    """
    Pressure drop in Zeotrpic mixtures correlation. Transfer learning from
    the Sun and Mishima correlation and the most influential parameters for
    the pressure drop in microchannels.

    Parameters
    ----------
    x : np.array(4,n)
        inputs of the neural network where n is the number of cases.
        the inputs are:
            - surface roughness in meters
            - Inner diameter in meters
            - Reynolds of the liquid part
            - Sun and Mishima Correlation

    Returns
    -------
    y: np.array(1,n)
        pressure drop in the channel

    """
    
    # Input
    x_norm = {}
    x_norm["xoffset"] = params["in_xoffset"].T
    x_norm["gain"] = params["in_gain"].T
    x_norm["ymin"] = params["in_ymin"]
    
    # Layer 1
    b1 = params["b1"]
    W1 = params["W1"]
    # Layer 2
    b2 = params["b2"]
    W2 = params["W2"]

    # Output
    y_norm = {}
    y_norm["ymin"] = params["out_ymin"]
    y_norm["gain"] = params["out_gain"]
    y_norm["xoffset"] = params["out_xoffset"]
    
    
    xp1 = mapminmax_apply(x, x_norm);
    a1 = tansig(W1 @ xp1.T + b1)
    a2 = W2 @ a1 + b2
    y = mapminmax_reverse(a2, y_norm)
    return y.T



