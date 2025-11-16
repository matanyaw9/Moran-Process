import tkinter as tk
from tkinter import ttk
import math
from moran_core import MoranProcess  # your logic file

class MoranGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Moran Process")
        self.process = None
        self.running = False
        self.step = 0

        # controls (N, fitness, etc.)
        # + canvas for grid
        # + start/stop/reset
        # + self.root.after(...) loop

if __name__ == "__main__":
    root = tk.Tk()
    app = MoranGUI(root)
    root.mainloop()
