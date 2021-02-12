import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure

def graph(tau, T, data_H = None, data_T = None, data_C= None, Hdata_H = None, Hdata_T = None, Hdata_C= None, diff_data_H = None, diff_data_T = None, diff_data_C= None):
    def animate(i):
        if i < data_H.shape[0]:
            ax[0][0].set_title(f"t = {str(round(i * tau, 1))} (days)")
            line_H.set_ydata(data_H[i])
            line_T.set_ydata(data_T[i])
            line_C.set_ydata(data_C[i])
            # ===
            Hline_H.set_ydata(Hdata_H[i])
            Hline_T.set_ydata(Hdata_T[i])
            Hline_C.set_ydata(Hdata_C[i])
            # ====
            diff_line_H.set_ydata(diff_data_H[i])
            diff_line_T.set_ydata(diff_data_T[i])
            diff_line_C.set_ydata(diff_data_C[i])
            # ====
            return line_H, line_T, line_C,\
                   Hline_H, Hline_T, Hline_C, \
                   diff_line_H, diff_line_T, diff_line_C,
        else:
            return False

    fig, ax = plt.subplots(3, 2)
    fig.canvas.set_window_title('Graph')
    x = range(0, len(data_H[0]), 1)
    N = np.size(data_H, 0) - 1
    ax[0][0].plot(x, data_H[0], color='black', )
    line_H, = ax[0][0].plot(x, data_H[N], color='blue')
    ax[0][0].set_title(f't = {str(T)} (days)')
    #ax[0][0].set_xlabel('distance (m)')
    ax[0][0].set_ylabel('H')
    ax[0][0].set_ylim(data_H.min() - 1, data_H.max() + 1)

    ax[1][0].plot(x, data_T[0], color='black')
    line_T, = ax[1][0].plot(x, data_T[N], color='red')
    #ax[1][0].set_title('T')
    #ax[1][0].set_xlabel('distance (m)')
    ax[1][0].set_ylabel('T')
    ax[1][0].set_ylim(data_T.min() - 1, data_T.max() + 1)

    ax[2][0].plot(x, data_C[0], color='black')
    line_C, = ax[2][0].plot(x, data_C[N], color='green')
    #legend_text = plt.text(1, -1, f"t = {str(0 * 0)} (days)")
    #ax[2][0].set_title('C')
    #ax[2][0].set_xlabel('distance (m)')
    ax[2][0].set_ylabel('C')
    ax[2][0].set_ylim(data_C.min() - 1, data_C.max() + 1)

# =====================

    #ax[0][0].plot(x, Hdata_H[0], color='black', )
    Hline_H, = ax[0][0].plot(x, Hdata_H[N], color='teal', linestyle='--')

    #ax[1][0].plot(x, Hdata_T[0], color='black')
    Hline_T, = ax[1][0].plot(x, Hdata_T[N], color='teal', linestyle='--')


    #ax[2][0].plot(x, Hdata_C[0], color='black')
    Hline_C, = ax[2][0].plot(x, Hdata_C[N], color='teal', linestyle='--')


# =====================

    #ax[0][1].plot(x, diff_data_H[0], color='black', )
    diff_line_H, = ax[0][1].plot(x, diff_data_H[N], color='blue')
    #ax[0][1].set_xlabel('distance (m)')
    ax[0][1].set_ylabel('Diff u')
    ax[0][1].set_ylim(diff_data_H.min(), diff_data_H.max())

    #ax[1][1].plot(x, diff_data_T[0], color='black')
    diff_line_T, = ax[1][1].plot(x, diff_data_T[N], color='red')
    #ax[1][1].set_title('T')
    #ax[1][1].set_xlabel('distance (m)')
    ax[1][1].set_ylabel('Diff T')
    ax[1][1].set_ylim(diff_data_T.min(), diff_data_T.max())

    #ax[2][0].plot(x, diff_data_C[0], color='black')
    diff_line_C, = ax[2][1].plot(x, diff_data_C[N], color='green')
    #legend_text = plt.text(1, -1, f"t = {str(0 * 0)} (days)")
    #ax[2][1].set_title('C')
    #ax[2][1].set_xlabel('distance (m)')
    ax[2][1].set_ylabel('Diff C')
    ax[2][1].set_ylim(diff_data_C.min(), diff_data_C.max())

# =====================

    fig.set_size_inches(14, 8)
    #legend_text = plt.text(5, 5, f"t = {str(0 * 0)} (days)")
    ani = animation.FuncAnimation(fig, animate, interval=200, save_count=data_H.shape[0]+1, blit=False)
    #ani.save('temp.gif', writer='imagemagick', fps=15)
    plt.tight_layout()
    plt.show()
