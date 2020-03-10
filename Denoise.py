
import pywt
from statsmodels.robust import mad

##################################################################################
# Denoising techniques
##################################################################################

#---------------------------------------------------------------------------------
def sub_rolling_mean(data):
    df_log = np.log(data)
    rolling_mean = df_log.rolling(window=12).mean()
    df_log_minus_mean = df_log - rolling_mean
    return df_log_minus_mean.dropna(inplace=True)

#---------------------------------------------------------------------------------
def mean_absolute_deviation(data, axis=None):
    """ Mean absolute deviation describes the randomnass in a signal
    """
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


 #---------------------------------------------------------------------------------
def wavelet_denoise(data, wavelet='db4', level=1):
    """ Calculate wavelet coefficience and use them to determine which values to keep
    """
    coeff = pywt.wavedec(data, wavelet, mode="per")
    sigma = (1/0.6745) * mean_absolute_deviation(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(data)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    return pywt.waverec(coeff, wavelet, mode='per')

#-----------------------------------------------------------------------------------
def average_denoise(signal, kernel_size=3, stride=1):
    """ Use simple moving average to denoise.
    """
    sample = []
    start = 0
    end = kernel_size
    while end <= len(signal):
        start = start + stride
        end = end + stride
        sample.extend(np.ones(end - start)*np.mean(signal[start:end]))
    return np.array(sample)

