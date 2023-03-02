import sys, os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pyaudio
import keyboard
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

from python_speech_features import logfbank
from queue import Queue

import noisereduce as nr

commands = [
  'backward',
  'bed',
  'bird',
  'cat',
  'dog',
  'down',
  'eight',
  'five',
  'follow',
  'forward',
  'four',
  'go',
  'happy',
  'house',
  'learn',
  'left',
  'marvin',
  'nine',
  'no',
  'off',
  'on',
  'one',
  'right',
  'seven',
  'sheila',
  'six',
  'stop',
  'three',
  'tree',
  'two',
  'up',
  'visual',
  'wow',
  'yes',
  'zero'
]

class_to_label = {i:commands[i] for i in range(len(commands))}
label_to_class = {commands[i]:i for i in range(len(commands))}

stop_by_voice = True

class DemoKeywordSpotting:
    def __init__(self, model_path, prob_threshold=0.9): 
        # load pre-trained model
        self.model = self.load_model(model_path)
        
        # parameters for input audio 
        self.CHUNK_DURATION = 0.9           # size of the chunck window, in seconds (select a fraction of second!)
        self.FORMAT = pyaudio.paInt16       # 16bit format per sample
        self.CHANNELS = 1                   # single channel for microphone
        self.RATE = 16000                   # samples per second [Hz] (common choice)
        self.INDEX = 0                      # audio device index
        self.CHUNK = int(self.CHUNK_DURATION * self.RATE)   # how many audio samples per frame we get 
        
        # windows parameters
        self.window_duration = 1            
        self.window_size = int(self.window_duration * self.RATE)
        
        # initialize window and data
        self.queue = Queue()
        self.data = np.zeros(self.window_size, dtype='int16')
        
        # classification threshold
        self.classification_threshold = prob_threshold

        # last two keywords detected
        self.last_2_kw = ['','']

        
    def stream(self):
        '''
        Streaming pipeline
        '''
        p, pystream = self.define_stream_object()
        pystream.start_stream()
        print('Process started...\n')
        
        plt.figure(figsize=(12, 8))
        
        try:
            while True:
                # get data
                data = self.queue.get()
                # if self.CHUNK_DURATION = 1 second, the import can be done without queueing as follows:
                # data = pystream.read(self.CHUNK)
                # data = np.frombuffer(data, dtype=np.int16)
                
                # band-pass filter
                use_bandpass = False
                if use_bandpass:
                    data = nr.reduce_noise(y=data, sr=self.RATE)
                      
                # get logfbanks
                data_fb = logfbank(
                    data,
                    samplerate = self.RATE, # samplerate of the signal we are working with
                    winlen     = 25/1000,   # length of the analysis window (milliseconds)
                    winstep    = 10/1000,   # step between successive windows (milliseconds)
                    nfilt      = 40,        # number of filters in the filterbank
                    nfft       = 512,       # FFT size
                    lowfreq    = 300,       # lowest band edge of mel filters (Hz)
                    highfreq   = None,      # highest band edge of mel filters (Hz)
                ).T
                
                # detect feature
                is_keyword, i_kw, kw, kw_prob, probabilities = self.spot_keyword(data_fb)
                
                # plot waveform and probabilities
                self.plot_input(data, i_kw, kw_prob, probabilities)
                
                # print results
                if is_keyword:
                    print('\x1b[1;32;40m' + 'KW: %s \nProbability %.1f%%' %(kw, kw_prob*100) + '\x1b[0m\n')
                    self.last_2_kw.pop(0)
                    self.last_2_kw.append(kw)
                    
                # stop with vocal command
                if stop_by_voice is True:
                    if ' '.join(self.last_2_kw) == 'off stop':
                        print('Process closed')
                        pystream.stop_stream()  # stop stream
                        pystream.close()        # close stream
                        p.terminate()           # release PortAudio system resources

                        try:
                            sys.exit(0)
                        except SystemExit:
                            os._exit(0)
                    
                                                 
        except KeyboardInterrupt:
            print('Process closed')
            pystream.stop_stream()  # stop stream
            pystream.close()        # close stream
            p.terminate()           # release PortAudio system resources
            
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
                
    
    def plot_input(self, data, i_kw, kw_prob, probabilities):
        
        plt.clf()
        
        # waveform
        plt.subplot(211)
        ax = plt.gca()
        ax.plot(data, color='black')
        ax.grid(alpha=0.5)
        ax.set_title("Input signal", fontsize=18)
        ax.set_ylabel("Amplitude", fontsize=15)
        ax.set_ylim(bottom=-35000, top=35000)
        ax.xaxis.set_major_locator(plt.NullLocator())
        
        # histogram    
        plt.subplot(212)    
        ax = plt.gca()
        bars = ax.bar(commands, probabilities, edgecolor='black', alpha=0.8, color='cornflowerblue')
        
        if kw_prob>self.classification_threshold:
            bars[i_kw].set_color('forestgreen')
            bars[i_kw].set_edgecolor('black')
            
        ax.grid(alpha=0.5)
        ax.set_title('Probability of each word', fontsize=18)
        ax.set_ylabel('Probability', fontsize=15)
        ax.set_ylim(bottom=0, top=100)
        ax.tick_params(axis='x', labelrotation=90)
        xmin, xmax = plt.xlim()
        ax.hlines(self.classification_threshold*100, xmin=xmin, xmax=xmax, linestyle='dashed', color='firebrick', linewidth=2)
        
        plt.tight_layout()
        plt.pause(.01)
    
              
    def spot_keyword(self, input):
        '''
        Spot a keyword in the current chunk
        '''
        # input = np.expand_dims(input, axis=0)
        input = tf.reshape(input, (1, 99, 40))
        prediction = self.model.predict(input, verbose=0).reshape(-1)
        
        i_kw = np.argmax(prediction)
        # print(class_to_label[i_kw], prediction[i_kw])
        
        if prediction[i_kw]>self.classification_threshold:
            is_kw = True
            output = class_to_label[i_kw]
            
        else:
            is_kw = False
            output = None
              
        return is_kw, i_kw, output, prediction[i_kw], prediction*100
        
        
    def define_stream_object(self):
        '''
        Input stream from the microphone
        '''
        # instantiate PyAudio and initialize PortAudio system resources
        p = pyaudio.PyAudio()
        
        # open stream
        stream = p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=self.INDEX, 
            stream_callback = self.queue_callback
        )
        
        return p, stream
    
    
    def queue_callback(self, in_data, frame_count, time_info, flag):
        '''
        This callback is needed to put the data in a queue, so that overlapping windows can be generated
        '''
        
        # get data
        in_data = np.frombuffer(in_data, dtype=np.int16)
        
        # update self.data
        self.data = np.append(self.data, in_data)
        
        # cut self.data to keep only the last window
        if len(self.data)>self.window_size:
            self.data = self.data[-self.window_size:]
            self.queue.put(self.data)
            
        return in_data, pyaudio.paContinue
        
        
    def reset_index(self, input_index):
        self.INDEX = int(input_index)
        

    @staticmethod 
    def load_model(model_path):
        ''' 
        Load the pre-trained model 
        '''
        pretrained_model = tf.keras.models.load_model(model_path)
        print(f'\nThe model "{model_path}" has been loaded!')
        
        return pretrained_model

    
if __name__=="__main__":
    models_path = "../trained_models"

    # get available models
    models_dict = {}
    for file in os.listdir(models_path):
        if file.endswith(".h5"):
            models_dict[len(models_dict)] = file
    
    # print available models
    print("\nAvailable models:")
    for key, value in models_dict.items():
        print(key, ':', value)

    # input model
    model_index = input('\n- Insert input model index and press Enter to continue\n... ')
    # print("")
    model_name = models_dict[int(model_index)]

    ks_streaming = DemoKeywordSpotting(os.path.join(models_path, model_name).replace("\\","/"))
    
    # input device
    device = input(
        '\n- Insert input device index and press Enter to continue\n- Insert -1 to see the list of the available input devices\n... '
    )
    
    if device == '-1':
        print('\nAvailable input devices:')
        print(sd.query_devices()) 
        device = input('\nInsert input device index and press Enter to continue... ')
        
    ks_streaming.reset_index(device)
    
    # streaming
    print("\nPress Enter to start streaming...")
    while True:
        try:
            if keyboard.is_pressed('enter'):
                ks_streaming.stream()
            
        except KeyboardInterrupt:
            print('\nKeyboard Interrupt!')
            
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)            

