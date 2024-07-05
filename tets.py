from CTkMessagebox import CTkMessagebox
from PIL import Image
from CTkToolTip import CTkToolTip
import customtkinter as ctk
from customtkinter import filedialog
import pandas as pd
ctk.set_appearance_mode("dark")
import lmfit
from functools import partial
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax import jit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

TITLE_COLOR = "#d08770"
ENABLE_COLOR = "#a3be8c"
DISABLE_COLOR = "#bf616a"
DESCRIPTION_COLOR = "#ebcb8b"

class FitParameter:
    def __init__(self, name: str, units: str | None, vary: bool, description: str):
        self.name = name
        self.units = units
        self.vary = vary
        self.description = description
        
    def set_entries_and_labels(self, frame: ctk.CTkFrame) -> None:
        self.init_entry = ctk.CTkEntry(frame, corner_radius=0, state="normal" if self.vary else "disabled")
        self.lb_entry = ctk.CTkEntry(frame, corner_radius=0, state="normal" if self.vary else "disabled")
        self.ub_entry = ctk.CTkEntry(frame, corner_radius=0, state="normal" if self.vary else "disabled")
        self.fixed_entry = ctk.CTkEntry(frame, corner_radius=0, state="disabled" if self.vary else "normal")
        self.init_label = ctk.CTkLabel(frame, text="init value", text_color=ENABLE_COLOR if self.vary else DISABLE_COLOR)
        self.lb_label = ctk.CTkLabel(frame, text="min", text_color=ENABLE_COLOR if self.vary else DISABLE_COLOR)
        self.ub_label = ctk.CTkLabel(frame, text="max", text_color=ENABLE_COLOR if self.vary else DISABLE_COLOR)
        self.fixed_label = ctk.CTkLabel(frame, text="fixed value", text_color=ENABLE_COLOR if not self.vary else DISABLE_COLOR)

    def get_entries(self) -> None:
        par_varies = self.vary.get()
        self.init_value = float(self.init_entry.get()) if par_varies else None
        self.lb = float(self.lb_entry.get()) if par_varies else None
        self.ub = float(self.ub_entry.get()) if par_varies else None
        self.fixed_value = None if par_varies else float(self.fixed_entry.get())


class ToplevelWindow(ctk.CTkToplevel):
    def __init__(self, fig):
        super().__init__()
        canvas = FigureCanvasTkAgg(master=self, figure=fig) # Convert the Figure to a tkinter widget
        canvas.draw() # Draw the graph on the canvas
        canvas.get_tk_widget().pack(fill='both', expand=True) # Show the widget on the screen


class FileSelectFrame(ctk.CTkFrame):
    def __init__(self, master, padding):
        super().__init__(master)
        self.master = master
        self.master.valid_data = False # flag to make sure that data file was selected
        ctk.CTkLabel(self, text="SELECT DATA FILE", text_color=TITLE_COLOR).grid(row=0, column=0, **padding, sticky="we")
        ctk.CTkButton(self, text="Browse data file", command=self.browsefunc).grid(row=1, **padding)
        
        self.file_label = ctk.CTkLabel(self, text="No file selected", anchor="w")
        self.file_label.grid(row=1, column=1, **padding)

        ctk.CTkLabel(self, text="Example Header:", anchor="w").grid(row=2, **padding)
        my_image = ctk.CTkImage(dark_image=Image.open("example_data.png"),
                                  size=(500, 100))
        ctk.CTkLabel(self, image=my_image, text="", anchor="w").grid(row=3, columnspan=2, **padding, sticky="w")
    
    def browsefunc(self):
        filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if filename:
            self.file_label.configure(text=filename)
            self.df = pd.read_excel(filename)
            self.master.offsets = np.asarray(self.df["ppm"].to_numpy().astype(float))
            self.master.powers = np.asarray(list(self.df.columns[1:].str.extract(r"(\d+.\d+)", expand=False)), dtype=float)
            self.master.data = np.asarray(self.df.to_numpy().astype(float).T[1:])
            self.master.valid_data = True

class ConstantsFrame(ctk.CTkFrame):
    def __init__(self, master, padding):
        super().__init__(master)
        self.master = master

        ctk.CTkLabel(self, text="SET CONSTANTS", text_color=TITLE_COLOR).grid(**padding)
        B0_label = ctk.CTkLabel(self, text="B₀ [T]")
        B0_label.grid(column=0, row=1,  **padding, sticky="w")
        CTkToolTip(B0_label, message="Static field strength", alpha=0.9, text_color=DESCRIPTION_COLOR)
        self.master.B0_entry = ctk.CTkEntry(self, corner_radius=0)
        self.master.B0_entry.grid(column=1, row=1,  **padding)

        gamma_label = ctk.CTkLabel(self, text="γ [10⁶ rad⋅s⁻¹⋅T⁻¹]")
        gamma_label.grid(column=0, row=2, **padding, sticky="w")
        CTkToolTip(gamma_label, message="Gyromagnetic ratio", alpha=0.9, text_color=DESCRIPTION_COLOR)
        self.master.gamma_entry = ctk.CTkEntry(self, corner_radius=0)
        self.master.gamma_entry.grid(column=1, row=2, **padding)

        tp_label = ctk.CTkLabel(self, text="tₚ [s]")
        tp_label.grid(column=0, row=3, **padding, sticky="w")
        CTkToolTip(tp_label, message="Saturation pulse duration", alpha=0.9, text_color=DESCRIPTION_COLOR)
        self.master.tp_entry = ctk.CTkEntry(self, corner_radius=0)
        self.master.tp_entry.grid(column=1, row=3,  **padding)

class FitParsFrame(ctk.CTkFrame):
    def __init__(self, master, padding):
        super().__init__(master)
        self.master = master
        ctk.CTkLabel(self, text="SET FIT/STATIC PARAMETERS", text_color=TITLE_COLOR).grid(**padding)
        # R1a
        self.master.R1a = FitParameter(name="R1a", units="Hz", vary=False, description="Longitudinal relaxation rate of pool a")
        self.create_fit_par_widgets(self, self.master.R1a, row=1, padding=padding)
        # R2a
        self.master.R2a = FitParameter(name="R2a", units="Hz", vary=False, description="Transverse relaxation rate of pool a")
        self.create_fit_par_widgets(self, self.master.R2a, row=2, padding=padding)
        # dwa
        self.master.dwa = FitParameter(name="Δωa", units="ppm", vary=False, description="Larmor frequency of pool a relative to itself.\nShould be zero")
        self.create_fit_par_widgets(self, self.master.dwa, row=3, padding=padding)
        # R1b
        self.master.R1b = FitParameter(name="R1b", units="Hz", vary=True, description="Longitudinal relaxation rate of pool b")
        self.create_fit_par_widgets(self, self.master.R1b, row=4, padding=padding)
        # R2b
        self.master.R2b = FitParameter(name="R2b", units="Hz", vary=True, description="Transverse relaxation rate of pool b")
        self.create_fit_par_widgets(self, self.master.R2b, row=5, padding=padding)
        # k
        self.master.k = FitParameter(name="k", units="Hz", vary=True, description="Exchange rate from pool b to pool a")
        self.create_fit_par_widgets(self, self.master.k, row=6, padding=padding)
        # f
        self.master.f = FitParameter(name="f", units=None, vary=True, description=("Equilibrium magnetization of pool b relative to pool a."
                                                                         "\nRoughly equivalent to fraction of pool b in solution"))
        self.create_fit_par_widgets(self, self.master.f, row=7, padding=padding)
        # dwb
        self.master.dwb = FitParameter(name="Δωb", units="ppm", vary=True, description="Larmor frequency of pool b relative to pool a")
        self.create_fit_par_widgets(self, self.master.dwb, row=8, padding=padding)

        ## Submit button
        self.valid_entries = False # flag to make sure that all entries are filled.
        ctk.CTkButton(self, text="Submit", anchor="w", command=self.sumbit_entries).grid(column=2, row=9, padx=10, pady=50, sticky="we")
        
        ## Fit button
        ctk.CTkButton(self, text="Fit Spectra", anchor="w", command=self.fit_spectra).grid(column=3, row=9, padx=10, pady=50, sticky="we")
        
        ## Show results button
        self.toplevel_window = None
        ctk.CTkButton(self, text="Show Results", anchor="w", command=self.open_toplevel).grid(column=4, row=9, padx=10, pady=50, sticky="we")


    def create_fit_par_widgets(self, frame:ctk.CTkFrame, p: FitParameter, row: int, padding: dict):
        if p.units is None:
            p.par_name = ctk.CTkLabel(frame, text=f"{p.name}")
        else:
            p.par_name = ctk.CTkLabel(frame, text=f"{p.name} [{p.units}]")
        CTkToolTip(p.par_name, message=p.description, alpha=0.9, text_color=DESCRIPTION_COLOR)
        p.par_name.grid(column=0, row=row, **padding, sticky="w")
        p.set_entries_and_labels(frame)
        p.vary = ctk.BooleanVar(master=frame, value=p.vary)
        p.init_label.grid(column=2, row=row, **padding)
        p.init_entry.grid(column=3, row=row, **padding)
        p.lb_label.grid(column=4, row=row, **padding)
        p.lb_entry.grid(column=5, row=row, **padding)
        p.ub_label.grid(column=6, row=row, **padding)
        p.ub_entry.grid(column=7, row=row, **padding)
        p.fixed_label.grid(column=8, row=row, **padding)
        p.fixed_entry.grid(column=9, row=row, **padding)
        ctk.CTkCheckBox(frame, text="vary", variable=p.vary,
                        command=lambda : self.checkbox_event(
                            p.vary, p.fixed_entry, p.init_entry, p.lb_entry, p.ub_entry, p.init_label, p.lb_label, p.ub_label, p.fixed_label)
                        ).grid(column=1, row=row,  **padding)
    
    def checkbox_event(self, vary_widget, fix_value_entry, init_value_entry, lb_entry, ub_entry, init_label, lb_label, ub_label, fixed_label):
        if vary_widget.get():
            fix_value_entry.configure(state="disabled")
            init_value_entry.configure(state="normal")
            lb_entry.configure(state="normal")
            ub_entry.configure(state="normal")
            fixed_label.configure(text_color=DISABLE_COLOR)
            init_label.configure(text_color=ENABLE_COLOR)
            lb_label.configure(text_color=ENABLE_COLOR)
            ub_label.configure(text_color=ENABLE_COLOR)
        else:
            fix_value_entry.configure(state="normal")
            init_value_entry.configure(state="disabled")
            lb_entry.configure(state="disabled")
            ub_entry.configure(state="disabled")
            fixed_label.configure(text_color=ENABLE_COLOR)
            init_label.configure(text_color=DISABLE_COLOR)
            lb_label.configure(text_color=DISABLE_COLOR)
            ub_label.configure(text_color=DISABLE_COLOR)

    def open_toplevel(self):
            if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
                self.toplevel_window = ToplevelWindow(self.fig)  # create window if its None or destroyed
            else:
                self.toplevel_window.focus()  # if window exists focus it

    def fit_spectra(self) -> None:
        if not (self.valid_entries and self.master.valid_data):
            CTkMessagebox(title="Error", message="Please fill all required fields\nand select data file.", icon="warning", height=500, width=1000)

        params = lmfit.Parameters()
        for p in [self.master.R1a, self.master.R2a, self.master.dwa, self.master.R1b, self.master.R2b, self.master.k, self.master.f, self.master.dwb]:
            params.add(name=p.name, value=p.init_value if p.vary.get() else p.fixed_value, vary=p.vary.get(), min=p.lb, max=p.ub)
        
        def residuals(params: lmfit.Parameters, offsets, powers, B0: float, gamma: float, tp: float, data):
            model_pars = np.array([params["R1a"], params["R2a"], params["Δωa"], params["R1b"], params["R2b"], params["k"], params["f"], params["Δωb"]])
            return  (data - bloch_mcconnell(model_pars, offsets, powers, B0, gamma, tp)).flatten()
        

        # fit = lmfit.minimize(residuals, params, args=(self.offsets, self.powers, self.B0, self.gamma, self.tp, self.data), method="differential_evolution", fit_kws={'seed': 0})
        fit = lmfit.minimize(fcn=residuals, params=params, method="COBYLA",
                             args=(self.master.offsets, self.master.powers, self.master.B0, self.master.gamma, self.master.tp, self.master.data))
        with open("fit.txt", "w") as text_file:
            text_file.write(lmfit.printfuncs.fit_report(fit))

        fit_pars = np.asarray(
            [fit.params[f"{p.name}"].value for p in [self.master.R1a, self.master.R2a, self.master.dwa, self.master.R1b, self.master.R2b, self.master.k, self.master.f, self.master.dwb]]
            )
        self.fig, ax = plt.subplots()
        ax.plot(self.master.offsets, bloch_mcconnell(fit_pars, self.master.offsets, self.master.powers, self.master.B0, self.master.gamma, self.master.tp).T)
        ax.set_prop_cycle(None)
        ax.plot(self.master.offsets, self.master.data.T, '.', label=[f"{power:.1f} μT" for power in self.master.powers])
        ax.set_xlabel("offset [ppm]")
        ax.set_ylabel("Z-value [a.u.]")
        ax.set_title("Nonlinear Least Squares Fit")
        ax.legend()
        self.fig.savefig("fit.pdf")

        CTkMessagebox(title="Info", message="Done!",
            icon="check", height=500, width=1000)
        
    def sumbit_entries(self):
        try:
            self.master.B0 = float(self.master.B0_entry.get())
            self.master.gamma = float(self.master.gamma_entry.get())
            self.master.tp = float(self.master.tp_entry.get())
            self.master.R1a.get_entries()
            self.master.R2a.get_entries()
            self.master.dwa.get_entries()
            self.master.R1b.get_entries()
            self.master.R2b.get_entries()
            self.master.k.get_entries()
            self.master.f.get_entries()
            self.master.dwb.get_entries()
            self.valid_entries = True
            CTkMessagebox(title="Info", message="Entries submitted successfully!\nClick 'Fit Spectra' to proceed.",
                        icon="check", height=500, width=1000)
        except:
            CTkMessagebox(title="Error", message="Please fill all required fields.", icon="warning", height=500, width=1000)

class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Z-Spectra Fitting Tool")
        self.minsize(1935, 1352)

        padding = {'padx':10, 'pady':10}
        file_select_frame = FileSelectFrame(self, padding)
        file_select_frame.pack(anchor="nw", **padding, fill="both", expand="true")

        # Frame to set constants
        constants_frame = ConstantsFrame(self, padding)
        constants_frame.pack(anchor="nw", **padding, fill="both", expand="true")

        # Frame to set fitting parameters and perform fit
        fit_pars_frame = FitParsFrame(self, padding)
        fit_pars_frame.pack(anchor="nw", **padding, fill="both", expand="true")


        # ############################## FOR TESTING PURPOSES ##############################
        self.B0_entry.insert(0, 7.4)
        self.gamma_entry.insert(0, 103.962)
        self.tp_entry.insert(0, 2.0)
        self.R1a.fixed_entry.insert(0, 8.0)
        self.R2a.fixed_entry.insert(0, 380)
        self.dwa.fixed_entry.insert(0, 0)
        self.R1b.lb_entry.insert(0, 0.1)
        self.R1b.ub_entry.insert(0, 100.0)
        self.R2b.lb_entry.insert(0, 1000)
        self.R2b.ub_entry.insert(0, 100_000)
        self.k.lb_entry.insert(0, 1)
        self.k.ub_entry.insert(0, 500)
        self.f.lb_entry.insert(0, 1e-5)
        self.f.ub_entry.insert(0, 0.1)
        self.dwb.lb_entry.insert(0, -265)
        self.dwb.ub_entry.insert(0, -255)
        for p in [self.R1b, self.R2b, self.k, self.f, self.dwb]:
            p.init_entry.insert(0, (float(p.lb_entry.get()) + float(p.ub_entry.get()))/2)
        #self.df = pd.read_excel("data.xlsx")
        # #############################################################################


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

