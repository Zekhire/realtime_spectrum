import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import cv2


def nothing(x):
    pass



# settings
cv2.namedWindow("settings")
cv2.createTrackbar("amplification", "settings", 1000, 10000, nothing)
cv2.createTrackbar("divider",       "settings", 10,   100,   nothing)
cv2.createTrackbar("shift",         "settings", 0,    1,     nothing)
cv2.createTrackbar("half",          "settings", 0,    1,     nothing)



rainbow = cv2.imread("rainbow.jpg")
N = 256
thickness = 4
forgetting_factor    = 0.9
amplification_factor = 1.1

# amplification     = 1000
# divider           = 10

# shift = True
# half  = False

# if half:
#     data_fft_abs_values = np.zeros(N//2)
# else:


def color_styles(key, height, width):
    if key == "rainbow":
        return rainbow

# color_styles(key="rainbow")


def plot_bars(black, data_fft_abs, thickness, settings_dict, key="rainbow"):
    h, w, _ = black.shape

    for i in range(len(data_fft_abs)):
    
        bin_height = data_fft_abs[i]*settings_dict["amplification"]
        if settings_dict["shift"] and i == len(data_fft_abs)//2:
            bin_height /= settings_dict["divider"]
        elif i == 0: 
            bin_height /= settings_dict["divider"]
            
        start_point = (i*thickness, h-1)
        end_point   = ((i+1)*thickness, h-1-int(bin_height))
        image_style = color_styles(key, h, w)
        color       = np.squeeze(rainbow[0, i*thickness])
        black = cv2.rectangle(black, start_point, end_point, (int(color[0]), int(color[1]), int(color[2])) , -1)
        black = cv2.rectangle(black, start_point, end_point, (0,0,0) , 1)
    return black


def visualization(data_fft_abs, width=256*3, height=600, thickness=3, settings_dict={}):
    black = np.zeros((height, width, 3), np.uint8)
    black_fft = plot_bars(black, data_fft_abs, thickness, settings_dict)
    cv2.imshow("visualization", black_fft)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    else:
        return False


def get_setting_dictionary():
    settings_dict = {}

    amplification = cv2.getTrackbarPos("amplification", "settings")
    divider       = cv2.getTrackbarPos("divider", "settings")
    shift_        = cv2.getTrackbarPos("shift", "settings")
    half_         = cv2.getTrackbarPos("half", "settings")

    if shift_ == 0:
        shift = False
    else:
        shift = True

    if half_ == 0:
        half = False
    else:
        half = True
    
    if divider == 0:
        divider = 1
    
    settings_dict["amplification"] = amplification
    settings_dict["divider"] = divider
    settings_dict["shift"] = shift
    settings_dict["half"] = half

    return settings_dict



def fft_processing(data_fft_abs_values, data_a, CHUNK, settings_dict, shift_old):
    # Perform FFT

    if shift_old != settings_dict["shift"]:
        shift_old = settings_dict["shift"]
        data_fft_abs_values = np.fft.fftshift(data_fft_abs_values)

    data_fft = np.fft.fft(data_a, CHUNK)
    if settings_dict["shift"]:
        data_fft = np.fft.fftshift(data_fft)
    
    if settings_dict["half"]:
        data_fft = data_fft[:N//2]

    data_fft_abs = abs(data_fft)

    for i in range(len(data_fft_abs_values)):
        if data_fft_abs_values[i]*forgetting_factor > data_fft_abs[i]:
            data_fft_abs_values[i] *= forgetting_factor
        elif data_fft_abs_values[i]*amplification_factor < data_fft_abs[i] and data_fft_abs_values[i] > 0.2:
            data_fft_abs_values[i] *= amplification_factor
        else:
            data_fft_abs_values[i] = data_fft_abs[i]

    return data_fft_abs_values, shift_old



def realtime_spectrum(path_config):
    CHUNK = N
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 11025
    RECORD_SECONDS = 5

    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format = FORMAT,
                    channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    input_device_index = 1, # 0 microphone
                    frames_per_buffer = CHUNK)

    print("* recording")

    frames = []

    # Record audio
    # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

    settings_dict = get_setting_dictionary()
    shift_old_ = 1-settings_dict["shift"]
    
    if shift_old_ == 0:
        shift_old = False
    else:
        shift_old = True

    data_fft_abs_values = np.zeros(N)

    while True:
        # Get data from microphone
        data = stream.read(CHUNK)

        # Convert data to float
        data_a = np.frombuffer((data), dtype=np.int16)
        data_a = data_a.astype('float_')/(2**15)

        settings_dict = get_setting_dictionary()

        # FFT processing
        data_fft_abs_values, shift_old = fft_processing(data_fft_abs_values, data_a, CHUNK, settings_dict, shift_old)

        # FFT visualization
        if visualization(data_fft_abs_values, CHUNK*thickness, thickness=thickness, settings_dict=settings_dict):
            break
        
    
        # frames.append(data)

    print("* done recording")
    
    # Stop stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # # Save to file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


if __name__ == "__main__":
    path_config = "./config.json"
    realtime_spectrum(path_config)