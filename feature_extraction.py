import numpy as np
import progressbar

def extract_features(raw_data,window_size):
    output_data = None
    feature_funcs = [mean_absolute_value,
                     waveform_length,
                     zero_crossing,
                     slope_sign_changes,
                     mean,
                     root_mean_square,
                     willison_amplitude,
                     simple_square_integral,
                     variance,
                     hjorth_mobility,
                     hjorth_complexity]

    for k in range(raw_data.shape[0]-1):
        # only extract features if this is the end of a gesture
        if raw_data[k,-1] == raw_data[k+1,-1] or raw_data[k,-1] == 0:
            continue
        data_slice = raw_data[k-window_size:k,:]
        time_values = data_slice[:,0]
        for channel in range(1,8):
            channel_signal = data_slice[:,channel]
            features =[]
            for feature_func in feature_funcs:
                features += [feature_func(time_values,channel_signal)]

def hjorth_complexity(time,X):
    timep, Xp = deriv(time,X)
    if variance(time,X) == 0:
        raise Exception("Hjorth Mobility of signal is zero, will result in infinite Hjorth Complexity")
    return hjorth_mobility(timep,Xp)/hjorth_mobility(time,X)

def hjorth_mobility(time,X):
    timep, Xp = deriv(time,X)
    if variance(time,X) == 0:
        raise Exception("variance of signal is zero, will result in infinite Hjorth Mobility")
    return (variance(timep,Xp)/variance(time,X))**0.5

def deriv(time,X):
    Xp = np.zeros(shape=(X.shape[0]-1))
    for k in range(X.shape[0]-1):
        Xp[k] = (X[k+1]-X[k])/(time[k+1]-time[k])
    new_time = time[:-1]
    return new_time, Xp

def variance(time,X):
    V = 0
    T = 0
    m = mean(time,X)
    for k in range(X.shape[0]-1):
        dt = time[k+1]-time[k]
        T += dt
        V += 0.5*((X[k+1]-m)**2 + (X[k]-m)**2)*dt
    return V/T

def simple_square_integral(time,X):
    SSI = 0
    for k in range(X.shape[0]-1):
        dt = time[k+1]-time[k]
        SSI += 0.5*(X[k+1]**2+X[k]**2)*dt
    return SSI

def willison_amplitude(time,X,threshold=0):
    WA = 0
    for k in range(X.shape[0]-1):
        if abs(X[k+1]-X[k]) > threshold:
            WA += 1
    return WA/X.shape[0]

def root_mean_square(time,X):
    RMS = 0
    T = 0
    for k in range(X.shape[0]-1):
        dt = time[k+1]-time[k]
        T += dt
        RMS += 0.5*(X[k+1]**2+X[k]**2)*dt
    return (RMS/T)**0.5

def mean(time,X):
    mean = 0
    T =  0
    for k in range(X.shape[0]-1):
        dt = time[k+1]-time[k]
        T += dt
        mean += dt*(X[k]+X[k+1])/2
    mean *= 1/T
    return mean

def slope_sign_changes(time,X,threshold=0):
    SSC = 0
    for k in range(1,len(X)-1):
        if (X[k]-X[k-1])*(X[k]-X[k+1]) > threshold:
            SSC += 1
    return SSC / X.shape[0]

def zero_crossing(time,X,threshold=0):
    ZC = 0
    for k in range(len(X)-1):
        if (X[k] > 0 and X[k+1] < 0) or (X[k] < 0 and X[k+1] > 0 and abs(X[k]-X[k+1])>threshold):
            ZC += 1
    return ZC

def waveform_length(time,X):
    WL = 0
    for i in range(len(X)-1):
        WL += abs(X[i+1]-X[i])
    return WL

def mean_absolute_value(time,X):
    return (1/X.shape[0])*np.absolute(X).sum()