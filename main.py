from CTkMessagebox import CTkMessagebox
from PIL import Image
from CTkToolTip import CTkToolTip
import customtkinter as ctk
from customtkinter import filedialog
import pandas as pd
ctk.set_appearance_mode("dark")
import lmfit
import numdifftools
from functools import partial
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax import jit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class FitParameter:
    def __init__(self, name: str, units: str | None, vary: bool, description: str):
        self.name = name
        self.units = units
        self.vary = vary
        self.description = description
        
    def set_entries_and_labels(self, frame: ctk.CTkFrame) -> None:
        self.init_entry = ctk.CTkEntry(frame, corner_radius=0, state="normal" if self.vary else "disabled", width=100)
        self.lb_entry = ctk.CTkEntry(frame, corner_radius=0, state="normal" if self.vary else "disabled", width=100)
        self.ub_entry = ctk.CTkEntry(frame, corner_radius=0, state="normal" if self.vary else "disabled", width=100)
        self.fixed_entry = ctk.CTkEntry(frame, corner_radius=0, state="disabled" if self.vary else "normal", width=100)
        self.init_label = ctk.CTkLabel(frame, text="init value", text_color="#a3be8c" if self.vary else "#bf616a")
        self.lb_label = ctk.CTkLabel(frame, text="min", text_color="#a3be8c" if self.vary else "#bf616a")
        self.ub_label = ctk.CTkLabel(frame, text="max", text_color="#a3be8c" if self.vary else "#bf616a")
        self.fixed_label = ctk.CTkLabel(frame, text="fixed value", text_color="#a3be8c" if not self.vary else "#bf616a")

    def get_entries(self) -> None:
        par_varies = self.vary_p.get()
        self.init_value = float(self.init_entry.get()) if par_varies else None
        self.lb = float(self.lb_entry.get()) if par_varies else None
        self.ub = float(self.ub_entry.get()) if par_varies else None
        self.fixed_value = None if par_varies else float(self.fixed_entry.get())


class ToplevelWindow(ctk.CTkToplevel):
    def __init__(self, fig):
        super().__init__()

        canvas = FigureCanvasTkAgg(master=self, figure=fig) # Convert the Figure to a tkinter widget
        canvas.draw() # Draw the graph on the canvas?
        canvas.get_tk_widget().pack(fill='both', expand=True) # Show the widget on the screen

class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Z-Spectra Fitting Tool")
        self.geometry("2500x1500")

        padding = {'padx':10, 'pady':10}

        # Frame to let the user select data file.
        # Create "data" variable to store a pandas DataFrame
        file_select_frame = ctk.CTkFrame(self)
        file_select_frame.pack(anchor="nw", fill="both", expand=True, **padding)

        ctk.CTkLabel(file_select_frame, text="SELECT DATA FILE", text_color="#d08770").grid(row=0, column=0, **padding, sticky="we")
        ctk.CTkButton(file_select_frame, text="Browse data file", command=self.browsefunc).grid(row=1, **padding)
        
        self.selected_dir_label = ctk.CTkLabel(file_select_frame, text="No directory selected", anchor="w")
        self.selected_dir_label.grid(row=1, column=1, **padding)

        ctk.CTkLabel(file_select_frame, text="Example Header:", anchor="w").grid(row=2, **padding)
        my_image = ctk.CTkImage(dark_image=Image.open("example_data.png"),
                                  size=(200, 20))
        ctk.CTkLabel(file_select_frame, image=my_image, text="", anchor="w").grid(row=3, columnspan=2, **padding, sticky="w")

        # Frame to set constants
        const_frame = ctk.CTkFrame(self)
        const_frame.pack(anchor="nw", fill="both", expand=True, **padding)
        ctk.CTkLabel(const_frame, text="SET CONSTANTS", text_color="#d08770").grid(**padding)

        B0_label = ctk.CTkLabel(const_frame, text="B₀ [T]")
        B0_label.grid(column=0, row=1,  **padding, sticky="w")
        CTkToolTip(B0_label, message="Static field strength", alpha=0.9, text_color="#ebcb8b")
        self.B0_entry = ctk.CTkEntry(const_frame, corner_radius=0)
        self.B0_entry.grid(column=1, row=1,  **padding)

        gamma_label = ctk.CTkLabel(const_frame, text="γ [10⁶ rad⋅s⁻¹⋅T⁻¹]")
        gamma_label.grid(column=0, row=2, **padding, sticky="w")
        CTkToolTip(gamma_label, message="Gyromagnetic ratio", alpha=0.9, text_color="#ebcb8b")
        self.gamma_entry = ctk.CTkEntry(const_frame, corner_radius=0)
        self.gamma_entry.grid(column=1, row=2, **padding)

        tp_label = ctk.CTkLabel(const_frame, text="tₚ [s]")
        tp_label.grid(column=0, row=3, **padding, sticky="w")
        CTkToolTip(tp_label, message="Saturation pulse duration", alpha=0.9, text_color="#ebcb8b")
        self.tp_entry = ctk.CTkEntry(const_frame, corner_radius=0)
        self.tp_entry.grid(column=1, row=3,  **padding)
    
        # Frame to set fitting parameters
        pars_frame = ctk.CTkFrame(self)
        pars_frame.pack(anchor="nw", fill="both", expand=True, **padding)
        ctk.CTkLabel(pars_frame, text="SET FIT/STATIC PARAMETERS", text_color="#d08770").grid(**padding)

        # R1a
        self.R1a = FitParameter(name="R1a", units="Hz", vary=False, description="Longitudinal relaxation rate of pool a")
        self.create_fit_par_widgets(pars_frame, self.R1a, row=1, padding=padding)
        # R2a
        self.R2a = FitParameter(name="R2a", units="Hz", vary=False, description="Transverse relaxation rate of pool a")
        self.create_fit_par_widgets(pars_frame, self.R2a, row=2, padding=padding)
        # dwa
        self.dwa = FitParameter(name="Δωa", units="ppm", vary=False, description="Larmor frequency of pool a relative to itself.\nShould be zero")
        self.create_fit_par_widgets(pars_frame, self.dwa, row=3, padding=padding)
        # R1b
        self.R1b = FitParameter(name="R1b", units="Hz", vary=True, description="Longitudinal relaxation rate of pool b")
        self.create_fit_par_widgets(pars_frame, self.R1b, row=4, padding=padding)
        # R2b
        self.R2b = FitParameter(name="R2b", units="Hz", vary=True, description="Transverse relaxation rate of pool b")
        self.create_fit_par_widgets(pars_frame, self.R2b, row=5, padding=padding)
        # k
        self.k = FitParameter(name="k", units="Hz", vary=True, description="Exchange rate from pool b to pool a")
        self.create_fit_par_widgets(pars_frame, self.k, row=6, padding=padding)
        # f
        self.f = FitParameter(name="f", units=None, vary=True, description=("Equilibrium magnetization of pool b relative to pool a."
                                                                         "\nRoughly equivalent to fraction of pool b in solution"))
        self.create_fit_par_widgets(pars_frame, self.f, row=7, padding=padding)
        # dwb
        self.dwb = FitParameter(name="Δωb", units="ppm", vary=True, description="Larmor frequency of pool b relative to pool a")
        self.create_fit_par_widgets(pars_frame, self.dwb, row=8, padding=padding)

        ## frame to put operation buttons
        buttons_frame = ctk.CTkFrame(self)
        buttons_frame.pack(anchor="center", fill="both", expand=True, **padding)

        # Submit button
        self.valid_entries = False # flag to make sure that all entries are filled.
        ctk.CTkButton(buttons_frame, text="Submit", anchor="w", command=self.sumbit_entries).grid(column=0, row=0, padx=50, pady=10, sticky="ew")
        
        # Fit button
        ctk.CTkButton(buttons_frame, text="Fit Spectra", anchor="w", command=self.fit_spectra).grid(column=1, row=0, padx=50, pady=10, sticky="ew")
        
        # Show results button
        self.toplevel_window = None
        ctk.CTkButton(buttons_frame, text="Show Results", anchor="w", command=self.open_toplevel).grid(column=2, row=0, padx=50, pady=10, sticky="ew")


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
        self.df = pd.read_excel("data.xlsx")
        # #############################################################################


    def open_toplevel(self):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = ToplevelWindow(self.fig)  # create window if its None or destroyed
        else:
            self.toplevel_window.focus()  # if window exists focus it

    def fit_spectra(self) -> None:
        if not self.valid_entries:
            CTkMessagebox(title="Error", message="Please fill all required fields.", icon="warning", height=50, width=100)
        self.offsets = np.asarray(self.df["ppm"].to_numpy().astype(float))
        self.powers = np.asarray(list(self.df.columns[1:].str.extract(r"(\d+.\d+)", expand=False)), dtype=float)
        self.data = np.asarray(self.df.to_numpy().astype(float).T[1:])
        params = lmfit.Parameters()
        for p in [self.R1a, self.R2a, self.dwa, self.R1b, self.R2b, self.k, self.f, self.dwb]:
            params.add(name=p.name, value=p.init_value if p.vary_p.get() else p.fixed_value, vary=p.vary_p.get(), min=p.lb, max=p.ub)
        
        def residuals(params: lmfit.Parameters, offsets, powers, B0: float, gamma: float, tp: float, data):
            model_pars = np.array([params["R1a"], params["R2a"], params["Δωa"], params["R1b"], params["R2b"], params["k"], params["f"], params["Δωb"]])
            return  (data - bloch_mcconnell(model_pars, offsets, powers, B0, gamma, tp)).flatten()
        

        # fit = lmfit.minimize(residuals, params, args=(self.offsets, self.powers, self.B0, self.gamma, self.tp, self.data), method="differential_evolution", fit_kws={'seed': 0})
        fit = lmfit.minimize(fcn=residuals, params=params, method="COBYLA", args=(self.offsets, self.powers, self.B0, self.gamma, self.tp, self.data))
        lmfit.printfuncs.report_fit(fit)

        fit_pars = np.asarray([fit.params[f"{p.name}"].value for p in [self.R1a, self.R2a, self.dwa, self.R1b, self.R2b, self.k, self.f, self.dwb]])
        self.fig, ax = plt.subplots()
        ax.plot(self.offsets, bloch_mcconnell(fit_pars, self.offsets, self.powers, self.B0, self.gamma, self.tp).T)
        ax.set_prop_cycle(None)
        ax.plot(self.offsets, self.data.T, '.', label=[f"{power:.1f} μT" for power in self.powers])
        ax.set_xlabel("offset [ppm]")
        ax.set_ylabel("Z-value [a.u.]")
        ax.set_title("Nonlinear Least Squares Fit")
        ax.legend()
        self.fig.savefig("test.pdf")

        CTkMessagebox(title="Info", message="Done!",
            icon="check", height=00, width=100)
        

    def sumbit_entries(self):
        try:
            self.B0 = float(self.B0_entry.get())
            self.gamma = float(self.gamma_entry.get())
            self.tp = float(self.tp_entry.get())
            self.R1a.get_entries()
            self.R2a.get_entries()
            self.dwa.get_entries()
            self.R1b.get_entries()
            self.R2b.get_entries()
            self.k.get_entries()
            self.f.get_entries()
            self.dwb.get_entries()
            self.valid_entries = True
            CTkMessagebox(title="Info", message="Entries submitted successfully!\nClick 'Fit Spectra' to proceed.",
                        icon="check", height=50, width=100)
        except:
            CTkMessagebox(title="Error", message="Please fill all required fields.", icon="warning", height=50, width=100)
    
    def create_fit_par_widgets(self, frame:ctk.CTkFrame, p: FitParameter, row: int, padding: dict):
        if p.units is None:
            p.par_name = ctk.CTkLabel(frame, text=f"{p.name}")
        else:
            p.par_name = ctk.CTkLabel(frame, text=f"{p.name} [{p.units}]")
        CTkToolTip(p.par_name, message=p.description, alpha=0.9, text_color="#ebcb8b")
        p.par_name.grid(column=0, row=row, **padding, sticky="w")
        p.set_entries_and_labels(frame)
        p.vary_p = ctk.BooleanVar(master=frame, value=p.vary)
        p.init_label.grid(column=2, row=row, **padding)
        p.init_entry.grid(column=3, row=row, **padding)
        p.lb_label.grid(column=4, row=row, **padding)
        p.lb_entry.grid(column=5, row=row, **padding)
        p.ub_label.grid(column=6, row=row, **padding)
        p.ub_entry.grid(column=7, row=row, **padding)
        p.fixed_label.grid(column=8, row=row, **padding)
        p.fixed_entry.grid(column=9, row=row, **padding)
        ctk.CTkCheckBox(frame, text="vary", variable=p.vary_p,
                        command=lambda : self.checkbox_event(
                            p.vary_p, p.fixed_entry, p.init_entry, p.lb_entry, p.ub_entry, p.init_label, p.lb_label, p.ub_label, p.fixed_label)
                        ).grid(column=1, row=row,  **padding)
    
    def checkbox_event(self, vary_widget, fix_value_entry, init_value_entry, lb_entry, ub_entry, init_label, lb_label, ub_label, fixed_label):
        if vary_widget.get():
            fix_value_entry.configure(state="disabled")
            init_value_entry.configure(state="normal")
            lb_entry.configure(state="normal")
            ub_entry.configure(state="normal")
            fixed_label.configure(text_color="#bf616a")
            init_label.configure(text_color="#a3be8c")
            lb_label.configure(text_color="#a3be8c")
            ub_label.configure(text_color="#a3be8c")
        else:
            fix_value_entry.configure(state="normal")
            init_value_entry.configure(state="disabled")
            lb_entry.configure(state="disabled")
            ub_entry.configure(state="disabled")
            fixed_label.configure(text_color="#a3be8c")
            init_label.configure(text_color="#bf616a")
            lb_label.configure(text_color="#bf616a")
            ub_label.configure(text_color="#bf616a")

    def browsefunc(self):
        filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if filename:
            self.selected_dir_label.configure(text=filename)
            self.df = pd.read_excel(filename)

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

