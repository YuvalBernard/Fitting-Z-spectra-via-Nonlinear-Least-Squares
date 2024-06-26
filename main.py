import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename

import pandas as pd

class MyGUI:

    def __init__(self, root):

        root.title("Z-Spectra Fitting Tool")
        root.geometry("2500x1500")

        # Frame to let the user select data file.
        # Create "data" variable to store a pandas DataFrame
        file_select_frame = ttk.Frame(root)
        file_select_frame.pack(anchor="nw")

        ttk.Label(file_select_frame, text="SELECT DATA FILE").grid(padx=10, pady=20, sticky="we")
        browse_button = ttk.Button(file_select_frame, text="Browse data file", command=self.browsefunc)
        browse_button.grid(row=1, padx=10, pady=20)

        self.selected_dir_label = ttk.Label(file_select_frame, text="No directory selected")
        self.selected_dir_label.grid(row=1, column=1, padx=10, pady=20)

        ttk.Label(file_select_frame, text="SET PARAMETERS").grid(padx=10, pady=20, sticky="we")

        # Frame to set static parameters
        static_pars_frame = ttk.Frame(root)
        static_pars_frame.pack(anchor="nw", pady=20)

        #ttk.Label(static_pars_frame, text="SET STATIC PARAMETERS").grid()
        ttk.Label(static_pars_frame, text="B₀").grid(column=0, row=0, padx=20)
        self.B0_entry = ttk.Entry(static_pars_frame, width=10)
        self.B0_entry.grid(column=1, row=0, padx=20)
        ttk.Label(static_pars_frame, text="T").grid(column=2, row=0, padx=20, sticky="w")

        ttk.Label(static_pars_frame, text="γ").grid(column=0, row=1, padx=20)
        self.B0_entry = ttk.Entry(static_pars_frame, width=10)
        self.B0_entry.grid(column=1, row=1, padx=20)
        ttk.Label(static_pars_frame, text="10⁶ rad⋅s⁻¹⋅T⁻¹").grid(column=2, row=1, padx=20, sticky="w")

        ttk.Label(static_pars_frame, text="tₚ").grid(column=0, row=2, padx=12)
        self.B0_entry = ttk.Entry(static_pars_frame, width=10)
        self.B0_entry.grid(column=1, row=2, padx=20)
        ttk.Label(static_pars_frame, text="s").grid(column=2, row=2, padx=20, sticky="w")
        



    def browsefunc(self):
        filename = askopenfilename()
        if filename:
            self.selected_dir_label.config(text=filename)
            self.data = pd.read_excel(filename)



root = tk.Tk()
MyGUI(root)
root.mainloop()

