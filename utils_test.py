# %%
import utils
import numpy as np
import matplotlib.pyplot as plt
from twotube import *
from threetube import *
from fourtube import *
from fivetube import *
from glottal import *
from HPF import *
import sounddevice as sd


# %%
def down_sample(xin, sampling_rate, over_sampling_ratio, cutoff=15000):
    if over_sampling_ratio == 1:
        return xin  # return xin itself, it's not over sample.

    # simple down sampler by FFT
    y = np.fft.fft(xin)
    freq = np.fft.fftfreq(len(xin), 1 / sampling_rate)
    id0 = np.where(freq > cutoff)[0][0]
    id1 = len(xin) - id0
    y[id0 : int(id1 + 1)] = 0.0
    z = np.real(np.fft.ifft(y))
    return z.reshape(
        int(len(xin) / over_sampling_ratio),
        over_sampling_ratio,
    )[:, 0].copy()


# %%
def generate_waveform1(tube, glo, hpf, repeat_num=50):
    yg_repeat = glo.make_N_repeat(repeat_num=repeat_num)
    y2tm = tube.process(yg_repeat)
    yout = hpf.iir1(y2tm)
    return yout


# %%
# Length & Area value, from problems 3.8 in "Digital Processing of Speech Signals" by L.R.Rabiner and R.W.Schafer
#
# /a/
L1_a = 9.0  # set list of 1st tube's length by unit is [cm]
A1_a = 1.0  # set list of 1st tube's area by unit is [cm^2]
L2_a = 8.0  # set list of 2nd tube's length by unit is [cm]
A2_a = 7.0  # set list of 2nd tube's area by unit is [cm^2]

# /u/
L1_u = 10.0  # set list of 1st tube's length by unit is [cm]
A1_u = 7.0  # set list of 1st tube's area by unit is [cm^2]
L2_u = 7.0  # set list of 2nd tube's length by unit is [cm]
A2_u = 3.0  # set list of 2nd tube's area by unit is [cm^2]

# /o/: L3,A3 is  extend factor to /a/ connecting as /u/
L3_o = L2_a * (L2_u / L1_u)  # set list of 3rd tube's length by unit is [cm]
A3_o = A2_a * (A2_u / A1_u)  # set list of 3rd tube's area by unit is [cm^2]

# %%
over_sampling_ratio = 1
fs = 16000

# insatnce Two tube model example
tube_2 = Class_TwoTube(L1_a, L2_a, A1_a, A2_a, sampling_rate=fs * over_sampling_ratio)

# insatnce Three tube model example
tube_3 = Class_ThreeTube(
    L1_a, L2_a, L3_o, A1_a, A2_a, A3_o, sampling_rate=fs * over_sampling_ratio
)

# insatnce Four tube model example
tube_4p1 = Class_FourTube(
    3.0, 8.8, 1.5, 1.4, 1.0, 8.4, 237.0, 11.6, sampling_rate=fs * over_sampling_ratio
)

# insatnce Five tube model examples
tube_5p1 = Class_FiveTube(
    2.9,
    8.7,
    1.4,
    2.9,
    1.5,
    1.0,
    21.9,
    438.0,
    166.0,
    62.0,
    sampling_rate=fs * over_sampling_ratio,
)
tube_5p2 = Class_FiveTube(
    5.0,
    2.6,
    2.7,
    2.6,
    2.5,
    19.1,
    1.0,
    1.3,
    10.8,
    69.0,
    sampling_rate=fs * over_sampling_ratio,
)
tube_5p3 = Class_FiveTube(
    1.3,
    4.9,
    1.1,
    2.5,
    4.9,
    19.8,
    2.8,
    15.8,
    1.0,
    1.0,
    sampling_rate=fs * over_sampling_ratio,
)
tube_5p4 = Class_FiveTube(
    2.7,
    5.4,
    2.7,
    1.4,
    4.0,
    1.0,
    5.6,
    2.1,
    1.1,
    3.2,
    sampling_rate=fs * over_sampling_ratio,
)
tube_5p5 = Class_FiveTube(
    3.3,
    1.6,
    5.9,
    6.3,
    1.3,
    2.1,
    3.0,
    1.1,
    8.8,
    1.0,
    sampling_rate=fs * over_sampling_ratio,
)

# %%
tube = tube_5p4

# %%
y = utils.generateNonStationaryNoise(fs * 2, fs)
y2tm = tube.process(y)
yout = hpf.iir1(y2tm)
yout = yout / yout.max()
plt.plot(yout)

# %%
ft = 20 * np.log10(np.abs(np.fft.fft(yout)))[: int(len(yout) / 4)]
plt.plot(ft)
# %%
sd.play(yout, fs)

# %%
sd.stop()

# %%
import huffman as hm
import numpy as np
import matplotlib.pyplot as plt

numbers = np.random.normal(20, size=(100000,)).round(2)
data = numbers.tobytes()
encoded, tree = hm.huffman_encode(data)

# Pretty print the Huffman table
print(f"Symbol Code\n------ ----")
for k, v in sorted(hm.huffman_table(tree).items(), key=lambda x: len(x[1])):
    print(f"{k:<6} {v}")

# Print the bit pattern of the encoded data
# print("".join(hm._bits_from_bytes(encoded)))

# Encode then decode
decoded = hm.huffman_decode(*hm.huffman_encode(data))
numbers_decoded = np.frombuffer(decoded)

# print(numbers - numbers_decoded)
print("Error:", np.linalg.norm(numbers - numbers_decoded))

# %%
