import string
import wave
from pathlib import Path
import simpleaudio
from synth_args import process_commandline
from nltk.corpus import cmudict
import re
from datetime import date
import random

import numpy
import os

import synth

x = synth.Synth('diphones')

y = synth.Utterance('hello world')

z = y.get_phone_seq()

a = x.phones_to_diphones(z)

c = x.synthesise(a)

print(z)
print(a)
print(c.data)

print(x.diphones['ow-pau.wav'])

c.play()
print(c.rate)

print()



