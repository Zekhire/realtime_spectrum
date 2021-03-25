import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt


def realtime_spectrum(path_config):

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
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