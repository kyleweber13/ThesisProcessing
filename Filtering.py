from scipy.signal import butter, lfilter


def filter_signal(data, type, low_f, high_f, sample_f, filter_order):
    """Function that creates bandpass filter to ECG data.

    Required arguments:
    -data: 3-column array with each column containing one accelerometer axis
    -type: "lowpass", "highpass" or "bandpass"
    -low_f, high_f: filter cut-offs, Hz
    -sample_f: sampling frequency, Hz
    -filter_order: order of filter; integer
    """

    nyquist_freq = 0.5 * sample_f

    if type == "lowpass":
        low = low_f / nyquist_freq
        b, a = butter(filter_order, low, btype="lowpass")
        filtered_data = lfilter(b, a, data)

    if type == "highpass":
        high = high_f / nyquist_freq

        b, a = butter(filter_order, high, btype="highpass")
        filtered_data = lfilter(b, a, data)

    if type == "bandpass":
        low = low_f / nyquist_freq
        high = high_f / nyquist_freq

        b, a = butter(filter_order, [low, high], btype="bandpass")
        filtered_data = lfilter(b, a, data)

    return filtered_data
