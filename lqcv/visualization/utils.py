from copy import copy
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

# Settings
matplotlib.rc("font", **{"size": 11})
matplotlib.use("Agg")  # for writing to files only


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    from scipy.signal import butter, filtfilt

    # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype="low", analog=False)

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)  # forward-backward filter


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=""):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]["lr"])
    plt.plot(y, ".-", label="LR")
    plt.xlabel("epoch")
    plt.ylabel("LR")
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / "LR.png", dpi=200)
    plt.close()
