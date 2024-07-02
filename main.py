from CTkMessagebox import CTkMessagebox
from PIL import Image
import CTkToolTip
import customtkinter as ctk
from customtkinter import filedialog
import pandas as pd
ctk.set_appearance_mode("dark")
import numdifftools
import lmfit
from functools import partial
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax import jit
import matplotlib.pyplot as plt

class ToplevelWindow(ctk.CTkToplevel):
    def __init__(self):
        # super().__init__(master = master)
        self.title("New Window")
        self.geometry("200x200")
        label = ctk.CTkLabel(self, text ="This is a new Window")
        label.pack()

class Parameter:
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

class BlochMcConnellGUI:

    def __init__(self, root):
        root.title("Z-Spectra Fitting Tool")
        root.geometry("2500x1500")

        padding = {'padx':10, 'pady':10}

        # Frame to let the user select data file.
        # Create "data" variable to store a pandas DataFrame
        file_select_frame = ctk.CTkFrame(root)
        file_select_frame.pack(anchor="nw")

        ctk.CTkLabel(file_select_frame, text="SELECT DATA FILE", text_color="#d08770").grid(row=0, column=0, **padding, sticky="we")
        ctk.CTkButton(file_select_frame, text="Browse data file", command=self.browsefunc).grid(row=1, **padding)
        
        self.selected_dir_label = ctk.CTkLabel(file_select_frame, text="No directory selected", anchor="w")
        self.selected_dir_label.grid(row=1, column=1, **padding)

        ctk.CTkLabel(file_select_frame, text="Example Header:", anchor="w").grid(row=2, **padding)
        my_image = ctk.CTkImage(dark_image=Image.open("example_data.png"),
                                  size=(500, 100))
        ctk.CTkLabel(file_select_frame, image=my_image, text="", anchor="w").grid(row=3, columnspan=2, **padding, sticky="w")

        # Frame to set constants
        const_frame = ctk.CTkFrame(root)
        const_frame.pack(anchor="nw", **padding)
        ctk.CTkLabel(const_frame, text="SET CONSTANTS", text_color="#d08770").grid(**padding)

        B0_label = ctk.CTkLabel(const_frame, text="B₀ [T]")
        B0_label.grid(column=0, row=1,  **padding, sticky="w")
        CTkToolTip.CTkToolTip(B0_label, message="Static field strength", alpha=0.9, text_color="#ebcb8b")
        self.B0_entry = ctk.CTkEntry(const_frame, corner_radius=0)
        self.B0_entry.grid(column=1, row=1,  **padding)

        gamma_label = ctk.CTkLabel(const_frame, text="γ [10⁶ rad⋅s⁻¹⋅T⁻¹]")
        gamma_label.grid(column=0, row=2, **padding, sticky="w")
        CTkToolTip.CTkToolTip(gamma_label, message="Gyromagnetic ratio", alpha=0.9, text_color="#ebcb8b")
        self.gamma_entry = ctk.CTkEntry(const_frame, corner_radius=0)
        self.gamma_entry.grid(column=1, row=2, **padding)

        tp_label = ctk.CTkLabel(const_frame, text="tₚ [s]")
        tp_label.grid(column=0, row=3, **padding, sticky="w")
        CTkToolTip.CTkToolTip(tp_label, message="Saturation pulse duration", alpha=0.9, text_color="#ebcb8b")
        self.tp_entry = ctk.CTkEntry(const_frame, corner_radius=0)
        self.tp_entry.grid(column=1, row=3,  **padding)
    
        # Frame to set fitting parameters
        pars_frame = ctk.CTkFrame(root)
        pars_frame.pack(anchor="nw", **padding)
        ctk.CTkLabel(pars_frame, text="SET FIT/STATIC PARAMETERS", text_color="#d08770").grid(**padding)

        # R1a
        self.R1a = Parameter(name="R1a", units="Hz", vary=False, description="Longitudinal relaxation rate of pool a")
        self.create_fit_par_widgets(pars_frame, self.R1a, row=1, padding=padding)
        # R2a
        self.R2a = Parameter(name="R2a", units="Hz", vary=False, description="Transverse relaxation rate of pool a")
        self.create_fit_par_widgets(pars_frame, self.R2a, row=2, padding=padding)
        # dwa
        self.dwa = Parameter(name="Δωa", units="ppm", vary=False, description="Larmor frequency of pool a relative to itself.\nShould be zero")
        self.create_fit_par_widgets(pars_frame, self.dwa, row=3, padding=padding)
        # R1b
        self.R1b = Parameter(name="R1b", units="Hz", vary=True, description="Longitudinal relaxation rate of pool b")
        self.create_fit_par_widgets(pars_frame, self.R1b, row=4, padding=padding)
        # R2b
        self.R2b = Parameter(name="R2b", units="Hz", vary=True, description="Transverse relaxation rate of pool b")
        self.create_fit_par_widgets(pars_frame, self.R2b, row=5, padding=padding)
        # k
        self.k = Parameter(name="k", units="Hz", vary=True, description="Exchange rate from pool b to pool a")
        self.create_fit_par_widgets(pars_frame, self.k, row=6, padding=padding)
        # f
        self.f = Parameter(name="f", units=None, vary=True, description=("Equilibrium magnetization of pool b relative to pool a."
                                                                         "\nRoughly equivalent to fraction of pool b in solution"))
        self.create_fit_par_widgets(pars_frame, self.f, row=7, padding=padding)
        # dwb
        self.dwb = Parameter(name="Δωb", units="ppm", vary=True, description="Larmor frequency of pool b relative to pool a")
        self.create_fit_par_widgets(pars_frame, self.dwb, row=8, padding=padding)

        ## Submit button
        self.valid_entries = False # flag to make sure that all entries are filled.
        ctk.CTkButton(pars_frame, text="Submit", anchor="w", command=self.sumbit_entries).grid(column=0, row=9, **padding, sticky="w")
        
        ## Fit button
        ctk.CTkButton(pars_frame, text="Fit Spectra", anchor="w", command=self.fit_spectra, fg_color="green").grid(column=0, row=10, **padding, sticky="w")
        
        ## Show results button
        self.toplevel_window = None
        ctk.CTkButton(pars_frame, text="Show Results", anchor="w", command=self.open_toplevel, fg_color="green").grid(column=0, row=11, **padding, sticky="w")

        # ############################## FOR TESTING PURPOSES ##############################
        self.B0_entry.insert(0, 7.4)
        self.gamma_entry.insert(0, 103.962)
        self.tp_entry.insert(0, 2)
        self.R1a.fixed_entry.insert(0, 8.0)
        self.R2a.fixed_entry.insert(0, 380)
        self.dwa.fixed_entry.insert(0, 0)
        self.R1b.init_entry.insert(0, 50)
        self.R1b.lb_entry.insert(0, 0.1)
        self.R1b.ub_entry.insert(0, 100.0)
        self.R2b.init_entry.insert(0, 50_000)
        self.R2b.lb_entry.insert(0, 1000)
        self.R2b.ub_entry.insert(0, 100_000)
        self.k.init_entry.insert(0, 250)
        self.k.lb_entry.insert(0, 1)
        self.k.ub_entry.insert(0, 500)
        self.f.init_entry.insert(0, 0.05)
        self.f.lb_entry.insert(0, 0.00001)
        self.f.ub_entry.insert(0, 0.1)
        self.dwb.init_entry.insert(0, -260)
        self.dwb.lb_entry.insert(0, -265)
        self.dwb.ub_entry.insert(0, -255)
        self.df = pd.read_excel("data.xlsx")
        # #############################################################################



    def open_toplevel(self):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = ToplevelWindow(root)  # create window if its None or destroyed
        else:
            self.toplevel_window.focus()  # if window exists focus it

    def fit_spectra(self) -> None:
        if not self.valid_entries:
            CTkMessagebox(title="Error", message="Please fill all required fields.", icon="warning", height=500, width=1000)
        self.offsets = jnp.asarray(self.df["ppm"].to_numpy().astype(float))
        self.powers = jnp.asarray(list(self.df.columns[1:].str.extract(r"(\d+.\d+)", expand=False)), dtype=float)
        self.data = jnp.asarray(self.df.to_numpy().astype(float).T[1:])
        params = lmfit.Parameters()
        # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        for p in [self.R1a, self.R2a, self.dwa, self.R1b, self.R2b, self.k, self.f, self.dwb]:
            params.add(name=p.name, value=p.init_value if p.vary_p.get() else p.fixed_value, vary=p.vary_p.get(), min=p.lb, max=p.ub)
        
        def residuals(params, offsets, powers, B0, gamma, tp, data):
            model_pars = np.array([params["R1a"], params["R2a"], params["Δωa"], params["R1b"], params["R2b"], params["k"], params["f"], params["Δωb"]])
            return  (data - bloch_mcconnell(model_pars, offsets, powers, B0, gamma, tp)).flatten()
        

        fit = lmfit.minimize(residuals, params, args=(self.offsets, self.powers, self.B0, self.gamma, self.tp, self.data), method="differential_evolution", fit_kws={'seed': 0})
        # fit = lmfit.minimize(fcn=residuals, params=params, method="least_squares", args=(self.offsets, self.powers, self.B0, self.gamma, self.tp, self.data))
        lmfit.printfuncs.report_fit(fit)

        fit_pars = np.asarray([fit.params[f"{p.name}"].value for p in [self.R1a, self.R2a, self.dwa, self.R1b, self.R2b, self.k, self.f, self.dwb]])
        fig, ax = plt.subplots()
        ax.plot(self.offsets, bloch_mcconnell(fit_pars, self.offsets, self.powers, self.B0, self.gamma, self.tp).T)
        ax.set_prop_cycle(None)
        ax.plot(self.offsets, self.data.T, '.')
        fig.savefig("test.svg")

        CTkMessagebox(title="Info", message="Done!",
            icon="check", height=500, width=1000)

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
                        icon="check", height=500, width=1000)
        except:
            CTkMessagebox(title="Error", message="Please fill all required fields.", icon="warning", height=500, width=1000)
    
    def create_fit_par_widgets(self, frame:ctk.CTkFrame, p: Parameter, row: int, padding: dict):
        if p.units is None:
            p.par_name = ctk.CTkLabel(frame, text=f"{p.name}")
        else:
            p.par_name = ctk.CTkLabel(frame, text=f"{p.name} [{p.units}]")
        CTkToolTip.CTkToolTip(p.par_name, message=p.description, alpha=0.9, text_color="#ebcb8b")
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
            # save data DataFrame
            self.df = pd.read_excel(filename)

@jit
@partial(jnp.vectorize, excluded=[0,1,3,4,5], signature="()->(k)") # powers
@partial(jnp.vectorize, excluded=[0,2,3,4,5], signature="()->()") # offsets
def bloch_mcconnell(model_pars, offset, power, B0, gamma, tp):
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

root = ctk.CTk()
BlochMcConnellGUI(root)
root.mainloop()

