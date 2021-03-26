import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import cv2

# height, width = 600, 1024

# red    = (0, 0, 255)
# orange = (0, 127, 255)
# yellow = (0, 255, 255)
# green  = (0, 255, 0)
# blue   = (255, 0, 0)
# indigo = (130, 0, 75)
# violet = (238, 130, 238)
# rainbow = np.array([[red, orange, yellow, green, blue, indigo, violet]], dtype=np.uint8)
# rainbow = cv2.resize(rainbow, (width, height))
# rainbow = cv2.imread("rainbow.jpg")
# cv2.imshow("rainbow", rainbow)
# cv2.imshow("rainbow2", cv2.resize(rainbow[0:1, 16:17], (32, 32)))
# cv2.waitKey(0)
# # cv2.imwrite("rainbow.jpg", rainbow)
# exit()

rainbow = cv2.imread("rainbow.jpg")


def color_styles(key, height, width):
    if key == "rainbow":
        return rainbow

# color_styles(key="rainbow")


def plot_bars(black, data_fft_abs, thickness, key="rainbow"):
    h, w, _ = black.shape
    for i in range(len(data_fft_abs)):
        start_point = (i*thickness, h-1)
        end_point   = ((i+1)*thickness, h-1-int(100*data_fft_abs[i]))
        image_style = color_styles(key, h, w)
        color       = np.squeeze(rainbow[0, i*thickness])
        black = cv2.rectangle(black, start_point, end_point, (int(color[0]), int(color[1]), int(color[2])) , -1)
        black = cv2.rectangle(black, start_point, end_point, (0,0,0) , 1)
    return black



def visualization(data_fft, width=256*3, height=600, thickness=3):
    black = np.zeros((height, width, 3), np.uint8)

    data_fft_abs = abs(data_fft)
    black_fft = plot_bars(black, data_fft_abs, thickness)
    cv2.imshow("visualization", black_fft)



def realtime_spectrum(path_config):

    thickness = 16

    CHUNK = 64
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 11025//4
    RECORD_SECONDS = 5

    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index = 0,
                frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    # Record audio
    # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    while True:
        # Get data from microphone
        data = stream.read(CHUNK)

        # Convert data to float
        data_a = np.frombuffer((data), dtype=np.int16)
        data_a = data_a.astype('float_')/(2**15)

        # Perform FFT
        data_fft = np.fft.fft(data_a, CHUNK)
        visualization(data_fft, CHUNK*thickness, thickness=thickness)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # print(data_a)
        # print(np.abs(data_fft))
        # print(max((data_a)/(2**15)), len(data_a), type(data_a[0]))
        # print(max(data), len(data_a))
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