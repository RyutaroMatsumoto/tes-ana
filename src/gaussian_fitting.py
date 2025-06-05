import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.cm as cm
def gaussian_fit(data, num_gauss=5):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import matplotlib.cm as cm

    def multi_gaussian(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amp = params[i]
            cen = params[i + 1]
            wid = params[i + 2]
            y += amp * np.exp(-((x - cen) ** 2) / (2 * wid ** 2))
        return y

    # ヒストグラム
    counts, bins = np.histogram(data, bins=300)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 初期値
    p0 = []
    for i in range(num_gauss):
        p0 += [np.max(counts) / (i + 1), 0.4 * i, 0.1]

    # フィット
    popt, _ = curve_fit(multi_gaussian, bin_centers, counts, p0=p0)

    # プロット用
    x_fit = np.linspace(min(bin_centers), max(bin_centers), 1000)
    y_fit = multi_gaussian(x_fit, *popt)

    # プロット作成
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=300, alpha=0.5, label="Data")
    ax.plot(x_fit, y_fit, color='black', linewidth=2, label="Total fit")

    # 各成分
    colors = cm.viridis(np.linspace(0, 1, num_gauss))
    for i in range(num_gauss):
        amp = popt[3 * i]
        cen = popt[3 * i + 1]
        wid = popt[3 * i + 2]
        y = amp * np.exp(-((x_fit - cen) ** 2) / (2 * wid ** 2))
        ax.plot(x_fit, y, linestyle='--', color=colors[i], label=f"n = {i}")

    ax.set_xlabel("Pulse height [a.u]")
    ax.set_ylabel("Counts")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return plt
