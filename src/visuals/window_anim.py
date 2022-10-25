import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import animation


class WindowedAnimator(tk.Tk):
    def __init__(self, fig, update_func, sample_rate=30):
        super().__init__()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().grid(column=0, row=0)
        self.ani = animation.FuncAnimation(
            fig, update_func, interval=1000/sample_rate)