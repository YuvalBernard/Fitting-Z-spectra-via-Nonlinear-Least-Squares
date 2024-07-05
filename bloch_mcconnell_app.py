from CTkMessagebox import CTkMessagebox
from PIL import Image
from CTkToolTip import CTkToolTip
import customtkinter as ctk
from customtkinter import filedialog
import pandas as pd
import lmfit
from functools import partial
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax import jit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Optional

# Configurations
ctk.set_appearance_mode("dark")

# Constants
ENABLE_COLOR = ("#F9F9FA", "#343638")
DISABLE_COLOR = "#bf616a"
ENTRY_WIDTH = 100
PADDING = {'padx':10, 'pady':10}

class ModelParameter:
    def __init__(self, name: str, units: str | None, description: str, vary: Optional[bool]=None):
        self.name = name
        self.units = units
        self.vary = vary
        self.description = description

    def set_entries_and_labels(self, master: ctk.CTkFrame) -> None:
        if self.vary is not None:
            if self.units is None:
                self.label = ctk.CTkLabel(master, text=f"{self.name}")
            else:
                self.label = ctk.CTkLabel(master, text=f"{self.name} [{self.units}]")
            self.init_label = ctk.CTkLabel(master, text="init value", anchor="w")
            self.lb_label = ctk.CTkLabel(master, text="min", anchor="w")
            self.ub_label = ctk.CTkLabel(master, text="max", anchor="w")
            self.fixed_label = ctk.CTkLabel(master, text="fixed value", anchor="w")
            self.init_entry = ctk.CTkEntry(master, width=ENTRY_WIDTH, corner_radius=0, state="normal" if self.vary else "disabled", fg_color=ENABLE_COLOR if self.vary else DISABLE_COLOR)
            self.lb_entry = ctk.CTkEntry(master, width=ENTRY_WIDTH, corner_radius=0, state="normal" if self.vary else "disabled", fg_color=ENABLE_COLOR if self.vary else DISABLE_COLOR)
            self.ub_entry = ctk.CTkEntry(master, width=ENTRY_WIDTH, corner_radius=0, state="normal" if self.vary else "disabled", fg_color=ENABLE_COLOR if self.vary else DISABLE_COLOR)
            self.fixed_entry = ctk.CTkEntry(master, width=ENTRY_WIDTH, corner_radius=0, state="disabled" if self.vary else "normal", fg_color=DISABLE_COLOR if self.vary else ENABLE_COLOR)
        else:
            self.label = ctk.CTkLabel(master, text=f"{self.name} [{self.units}]")
            self.entry = ctk.CTkEntry(master, width=ENTRY_WIDTH, corner_radius=0)

    def get_entries(self) -> None:
        if self.vary is not None:
            par_varies = self.vary.get()
            self.init_value = float(self.init_entry.get()) if par_varies else None
            self.lb = float(self.lb_entry.get()) if par_varies else None
            self.ub = float(self.ub_entry.get()) if par_varies else None
            self.fixed_value = None if par_varies else float(self.fixed_entry.get())
        else:
            self.value = float(self.entry.get())

class FileSelectFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        
        self.select_file_frame = ctk.CTkFrame(self)
        self.select_file_frame.pack(fill="both", expand=True)
        ctk.CTkButton(self.select_file_frame, text="Browse data file", command=self.browsefunc).pack(**PADDING)
        self.file_path_label = ctk.CTkLabel(self.select_file_frame, text="Path: No file selected.", anchor="w")
        self.file_path_label.pack(**PADDING, anchor="w")
        self.file_offsets_label = ctk.CTkLabel(self.select_file_frame, text="Offsets on range:", anchor="w")
        self.file_offsets_label.pack(**PADDING, anchor="w")
        self.file_powers_label = ctk.CTkLabel(self.select_file_frame, text="Irradiation amplitudes:", anchor="w")
        self.file_powers_label.pack(**PADDING, anchor="w")

        table_frame = ctk.CTkFrame(self)
        ctk.CTkLabel(self, text="Example Table: (Header must have specific form)", anchor="w").pack(**PADDING)
        image = ctk.CTkImage(dark_image=Image.open("example_data.png"), size=(250, 200))
        ctk.CTkLabel(self, image=image, text="").pack(**PADDING)
    
    def browsefunc(self) -> None:
        filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if filename:
            self.df = pd.read_excel(filename)
            powers = list(self.df.columns[1:].str.extract(r"(\d+.\d+)", expand=False))
            self.master.offsets = np.asarray(self.df["ppm"].to_numpy().astype(float))
            self.master.powers = np.asarray(powers, dtype=float)
            self.master.data = np.asarray(self.df.to_numpy().astype(float).T[1:])
            self.file_path_label.configure(text=f"Path: {filename}")
            self.file_offsets_label.configure(text=f"Offsets on range: {self.master.offsets[0]} to {self.master.offsets[-1]} ppm")
            self.file_powers_label.configure(text=f"Irradiation amplidtudes: {', '.join(powers)} µT")


class ConstantsFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        ctk.CTkLabel(self, text="Tip: Hover over parameter label for a short description").pack(**PADDING)
        
        self.master.B0 = ModelParameter(name="B₀", units="T", description="Static field strength")
        self.master.gamma = ModelParameter(name="γ", units="10⁶ rad⋅s⁻¹⋅T⁻¹", description="Gyromagnetic ratio")
        self.master.tp = ModelParameter(name="tₚ", units="s", description="Saturation pulse duration")

        self.pars_frame = ctk.CTkFrame(self)
        self.pars_frame.pack(**PADDING)
        for row, p in enumerate([self.master.B0, self.master.gamma, self.master.tp]):
            self.create_fit_par_widgets(p, row)

        # for testing purposes. remove in main version
        self.master.B0.entry.insert(0, 7.4)
        self.master.gamma.entry.insert(0, 103.962)
        self.master.tp.entry.insert(0, 2.0)
        

    def create_fit_par_widgets(self, p: ModelParameter, row: int) -> None:
        p.set_entries_and_labels(self.pars_frame)
        p.label.grid(column=0, row=row, **PADDING)
        CTkToolTip(p.label, message=p.description, alpha=0.9)
        p.entry.grid(column=1, row=row, **PADDING)


class FitParsFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        ctk.CTkLabel(self, text="Tip: Hover over parameter label for a short description").pack(**PADDING)

        self.master.R1a = ModelParameter(name="R1a", units="Hz", vary=False, description="Longitudinal relaxation rate of pool a")
        self.master.R2a = ModelParameter(name="R2a", units="Hz", vary=False, description="Transverse relaxation rate of pool a")
        self.master.dwa = ModelParameter(name="Δωa", units="ppm", vary=False, description="Larmor frequency of pool a relative to itself.\nShould be zero")
        self.master.R1b = ModelParameter(name="R1b", units="Hz", vary=True, description="Longitudinal relaxation rate of pool b")
        self.master.R2b = ModelParameter(name="R2b", units="Hz", vary=True, description="Transverse relaxation rate of pool b")
        self.master.k = ModelParameter(name="k", units="Hz", vary=True, description="Exchange rate from pool b to pool a")
        self.master.f = ModelParameter(name="f", units=None, vary=True, description=("Equilibrium magnetization of pool b relative to pool a."
                                                                         "\nRoughly equivalent to fraction of pool b in solution"))
        self.master.dwb = ModelParameter(name="Δωb", units="ppm", vary=True, description="Larmor frequency of pool b relative to pool a")

        self.pars_frame = ctk.CTkFrame(self)
        self.pars_frame.pack(**PADDING)
        for row, p in enumerate([self.master.R1a, self.master.R2a, self.master.dwa, self.master.R1b, self.master.R2b, self.master.k, self.master.f, self.master.dwb]):
            self.create_fit_par_widgets(p, row)

        # for testing purposes. remove in main version
        self.master.R1a.fixed_entry.insert(0, 8.0)
        self.master.R2a.fixed_entry.insert(0, 380)
        self.master.dwa.fixed_entry.insert(0, 0)
        self.master.R1b.lb_entry.insert(0, 0.1)
        self.master.R1b.ub_entry.insert(0, 100.0)
        self.master.R2b.lb_entry.insert(0, 1000)
        self.master.R2b.ub_entry.insert(0, 100_000)
        self.master.k.lb_entry.insert(0, 1)
        self.master.k.ub_entry.insert(0, 500)
        self.master.f.lb_entry.insert(0, 1e-5)
        self.master.f.ub_entry.insert(0, 0.1)
        self.master.dwb.lb_entry.insert(0, -265)
        self.master.dwb.ub_entry.insert(0, -255)
        for p in [self.master.R1b, self.master.R2b, self.master.k, self.master.f, self.master.dwb]:
            p.init_entry.insert(0, (float(p.lb_entry.get()) + float(p.ub_entry.get()))/2)

    def create_fit_par_widgets(self, p: ModelParameter, row: int) -> None:
        p.set_entries_and_labels(self.pars_frame)
        p.label.grid(column=0, row=row, **PADDING)
        CTkToolTip(p.label, message=p.description, alpha=0.9)
        p.vary = ctk.BooleanVar(value=p.vary)
        ctk.CTkCheckBox(self.pars_frame, text="vary", variable=p.vary, command=lambda : self.checkbox_event(p)).grid(column=1, row=row,  **PADDING)
        p.init_label.grid(column=2, row=row, **PADDING)
        p.init_entry.grid(column=3, row=row, **PADDING)
        p.lb_label.grid(column=4, row=row, **PADDING)
        p.lb_entry.grid(column=5, row=row, **PADDING)
        p.ub_label.grid(column=6, row=row, **PADDING)
        p.ub_entry.grid(column=7, row=row, **PADDING)
        p.fixed_label.grid(column=8, row=row, **PADDING)
        p.fixed_entry.grid(column=9, row=row, **PADDING)
    
    def checkbox_event(self, p: ModelParameter) -> None:
        if p.vary.get():
            p.fixed_entry.configure(state="disabled", fg_color=DISABLE_COLOR)
            p.init_entry.configure(state="normal", fg_color=ENABLE_COLOR)
            p.lb_entry.configure(state="normal", fg_color=ENABLE_COLOR)
            p.ub_entry.configure(state="normal", fg_color=ENABLE_COLOR)
        else:
            p.fixed_entry.configure(state="normal", fg_color=ENABLE_COLOR)
            p.init_entry.configure(state="disabled", fg_color=DISABLE_COLOR)
            p.lb_entry.configure(state="disabled", fg_color=DISABLE_COLOR)
            p.ub_entry.configure(state="disabled", fg_color=DISABLE_COLOR)

class ToplevelWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__()
        self.master = master
        
        fig, ax = plt.subplots()
        ax.plot(self.master.offsets, self.master.best_fit_spectra.T)
        ax.set_prop_cycle(None)
        ax.plot(self.master.offsets, self.master.data.T, '.', label=[f"{power:.1f} μT" for power in self.master.powers])
        ax.set_xlabel("offset [ppm]")
        ax.set_ylabel("Z-value [a.u.]")
        ax.set_title("Nonlinear Least Squares Fit")
        ax.legend()
        fig.savefig("fit.pdf")
        plt.close(fig)

        canvas = FigureCanvasTkAgg(master=self, figure=fig) # Convert the Figure to a tkinter widget
        canvas.draw() # Draw the graph on the canvas
        canvas.get_tk_widget().pack(fill='both', expand=True) # Show the widget on the screen

class MyTabView(ctk.CTkTabview):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        # create tabs
        self.add("Select Data")
        self.add("Set Constants")
        self.add("Set Fit Parameters")

        # add widgets on tabs
        file_select_frame = FileSelectFrame(self.tab("Select Data"))
        file_select_frame.pack()
        constants_frame = ConstantsFrame(self.tab("Set Constants"))
        constants_frame.pack()
        fit_pars_frame = FitParsFrame(self.tab("Set Fit Parameters"))
        fit_pars_frame.pack()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.CTkLabel(self, text="Welcome!\nFill required information in all tabs\nthen click 'Submit' and 'Fit Spectra'.\nClick 'Show Results' to view plot of fit.").pack()
        self.valid_entries = False # flag to make sure entries are submitted correctly.
        
        self.tab_view = MyTabView(self)
        self.tab_view.pack(fill="both", expand=True)
        buttons_frame = ctk.CTkFrame(self)

        buttons_frame.pack(anchor="center", fill="both", expand=True, pady=10)
        ctk.CTkButton(buttons_frame, width=75, text="Submit", command=self.sumbit_entries).pack(side="left", padx=10, expand=True)
        ctk.CTkButton(buttons_frame, width=75, text="Fit Spectra", command=self.fit_spectra).pack(side="left", padx=10, expand=True)
        ctk.CTkButton(buttons_frame, width=75, text="Show Fit", command=self.show_fit).pack(side="left", padx=10, expand=True)
        
        # flag to make sure that results windows is not displayed twice
        self.toplevel_window = None

    def sumbit_entries(self) -> None:
        ct = self.tab_view.tab("Set Constants") # to access attributes of constants tab
        ft = self.tab_view.tab("Set Fit Parameters") # to access attributes of fit pars tab
        dt = self.tab_view.tab("Select Data") # to access attributes of data selection tab
        try:
            for p in [ct.B0, ct.gamma, ct.tp, ft.R1a, ft.R2a, ft.dwa, ft.R1b, ft.R2b, ft.k, ft.f, ft.dwb]:
                p.get_entries()
            self.B0 = ct.B0.value
            self.gamma = ct.gamma.value
            self.tp = ct.tp.value
            self.R1a = ft.R1a
            self.R2a = ft.R2a
            self.dwa = ft.dwa
            self.R1b = ft.R1b
            self.R2b = ft.R2b
            self.k = ft.k
            self.f = ft.f
            self.dwb = ft.dwb
            self.offsets = dt.offsets
            self.powers = dt.powers
            self.data = dt.data
            self.valid_entries = True
            CTkMessagebox(title="Info", message="Entries submitted successfully!\nClick 'Fit Spectra' to proceed.",
                        icon="check", height=50, width=100)
        except:
            CTkMessagebox(title="Error", message="Please fill all required fields\nand select data file.", icon="warning", height=50, width=100)

    def fit_spectra(self) -> None:
        if not self.valid_entries:
            CTkMessagebox(title="Error", message="Please fill all required fields\nand select data file.", icon="warning", height=50, width=100)
            return

        params = lmfit.Parameters()
        for p in [self.R1a, self.R2a, self.dwa, self.R1b, self.R2b, self.k, self.f, self.dwb]:
            params.add(name=p.name, value=p.init_value if p.vary.get() else p.fixed_value, vary=p.vary.get(), min=p.lb, max=p.ub)
        
        def residuals(params: lmfit.Parameters, offsets, powers, B0, gamma, tp, data):
            model_pars = np.array([params["R1a"], params["R2a"], params["Δωa"], params["R1b"], params["R2b"], params["k"], params["f"], params["Δωb"]])
            return  (data - bloch_mcconnell(model_pars, offsets, powers, B0, gamma, tp)).flatten()
        

        # fit = lmfit.minimize(residuals, params, args=(self.offsets, self.powers, self.B0, self.gamma, self.tp, self.data), method="differential_evolution", fit_kws={'seed': 0})
        fit = lmfit.minimize(fcn=residuals, params=params, method="COBYLA",
                             args=(self.offsets, self.powers, self.B0, self.gamma, self.tp, self.data))
        with open("fit.txt", "w") as text_file:
            text_file.write(lmfit.printfuncs.fit_report(fit))

        self.best_fit_pars = np.asarray(
            [fit.params[f"{p.name}"].value for p in [self.R1a, self.R2a, self.dwa, self.R1b, self.R2b, self.k, self.f, self.dwb]]
            )
        self.best_fit_spectra = bloch_mcconnell(self.best_fit_pars, self.offsets, self.powers, self.B0, self.gamma, self.tp)

        CTkMessagebox(self, title="Info", message="Done!", icon="check", height=50, width=100)

    def show_fit(self):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = ToplevelWindow(self)  # create window if its None or destroyed
        else:
            self.toplevel_window.focus()  # if window exists focus it

@jit
@partial(jnp.vectorize, excluded=[0,1,3,4,5], signature="()->(k)") # powers
@partial(jnp.vectorize, excluded=[0,2,3,4,5], signature="()->()") # offsets
def bloch_mcconnell(model_pars, offset:float, power:float, B0:float, gamma:float, tp:float):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_pars
    dwa *= B0*gamma
    dwb *= B0*gamma
    offset *= B0*gamma
    power *= gamma
    ka = k * f
    M0 = jnp.array([0, 0, 1, 0, 0, f, 1])
    A = jnp.array([
        [-(R2a + ka), dwa-offset, 0, k, 0, 0, 0],
        [offset-dwa, -(R2a + ka), power, 0, k, 0, 0],
        [0, -power, -(R1a + ka), 0, 0, k, R1a],
        [ka, 0, 0, -(R2b + k), dwb-offset, 0, 0],
        [0, ka, 0, offset-dwb, -(R2b + k), power, 0],
        [0, 0, ka, 0, -power, -(R1b + k), R1b * f],
        [0, 0, 0, 0, 0, 0, 0]])
    Z = jnp.matmul(expm(A * tp, max_squarings=18), M0)[2]
    return Z

app = App()
app.mainloop()