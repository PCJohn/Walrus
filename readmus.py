import subprocess as sp
import numpy as np
from matplotlib import pyplot as plt

OUT_FREQ = 44100
command = ['ffmpeg',
          '-i','/mnt/share/music/The Beatles/Abbey Road/Mean Mr Mustard.mp3',
          '-f','s16le',
          '-acodec','pcm_s16le',
          '-ar',str(OUT_FREQ), #Output frequency = 44100
          '-ac','2', #Stereo - Set to 1 for mono
          '-']
pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
raw_audio = pipe.stdout.read(2*OUT_FREQ*512)
audio = np.fromstring(raw_audio,dtype='int16')
audio = audio.reshape((len(audio)/2,2))

print audio

#plt.plot(audio[:,0])
#plt.show()
