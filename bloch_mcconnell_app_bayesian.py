from CTkMessagebox import CTkMessagebox
from PIL import Image
from CTkToolTip import CTkToolTip
import customtkinter as ctk
from customtkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Optional
import os
from functools import partial
import numpy as np
import jax.numpy as jnp
from solve_bloch_mcconnell import Z_analytical_symbolic
from jax import jit, config, random
from jax.scipy.stats import norm
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
import arviz as az

# Configurations
ctk.set_appearance_mode("light")
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

# ctk.set_widget_scaling(2)
# ctk.set_window_scaling(2)

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
            self.min_label = ctk.CTkLabel(master, text="min", anchor="w")
            self.max_label = ctk.CTkLabel(master, text="max", anchor="w")
            self.fixed_label = ctk.CTkLabel(master, text="fixed value", anchor="w")
            self.min_entry = ctk.CTkEntry(master, width=ENTRY_WIDTH, corner_radius=0, state="normal" if self.vary else "disabled", fg_color=ENABLE_COLOR if self.vary else DISABLE_COLOR)
            self.max_entry = ctk.CTkEntry(master, width=ENTRY_WIDTH, corner_radius=0, state="normal" if self.vary else "disabled", fg_color=ENABLE_COLOR if self.vary else DISABLE_COLOR)
            self.fixed_entry = ctk.CTkEntry(master, width=ENTRY_WIDTH, corner_radius=0, state="disabled" if self.vary else "normal", fg_color=DISABLE_COLOR if self.vary else ENABLE_COLOR)
        else:
            self.label = ctk.CTkLabel(master, text=f"{self.name} [{self.units}]")
            self.entry = ctk.CTkEntry(master, width=ENTRY_WIDTH, corner_radius=0)

    def get_entries(self) -> None:
        if self.vary is not None:
            par_varies = self.vary.get()
            self.min = float(self.min_entry.get()) if par_varies else None
            self.max = float(self.max_entry.get()) if par_varies else None
            self.fixed_value = None if par_varies else float(self.fixed_entry.get())
        else:
            self.value = float(self.entry.get())

    def set_prior(self) -> None:
        def normal_dist_from_quantiles(x1, x2, p1, p2):
            mu = (x1*norm.ppf(p2) - x2*norm.ppf(p1))/(norm.ppf(p2) - norm.ppf(p1))
            sigma = (x2 - x1)/(norm.ppf(p2) - norm.ppf(p1))
            return dist.TruncatedNormal(mu, sigma, low=0)

        par_varies = self.vary.get()
        if par_varies:
            if self.name in ["Δωa", "Δωb"]:
                self.prior = dist.Uniform(self.min, self.max)
            elif self.name in ["R1a", "R2a", "R1b", "R2b", "k", "f"]:
                self.prior = normal_dist_from_quantiles(self.min, self.max, 0.025, 0.975)
            else:
                raise ValueError("B0, gamma and tp are constants and do not have priors.")
        else:
            self.prior = None

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
        self.master.R1b.min_entry.insert(0, 0.1)
        self.master.R1b.max_entry.insert(0, 100.0)
        self.master.R2b.min_entry.insert(0, 1000)
        self.master.R2b.max_entry.insert(0, 100_000)
        self.master.k.min_entry.insert(0, 1)
        self.master.k.max_entry.insert(0, 500)
        self.master.f.min_entry.insert(0, 1e-5)
        self.master.f.max_entry.insert(0, 0.1)
        self.master.dwb.min_entry.insert(0, -265)
        self.master.dwb.max_entry.insert(0, -255)

    def create_fit_par_widgets(self, p: ModelParameter, row: int) -> None:
        p.set_entries_and_labels(self.pars_frame)
        p.label.grid(column=0, row=row, **PADDING)
        CTkToolTip(p.label, message=p.description, alpha=0.9)
        p.vary = ctk.BooleanVar(value=p.vary)
        ctk.CTkCheckBox(self.pars_frame, text="vary", variable=p.vary, command=lambda : self.checkbox_event(p)).grid(column=1, row=row,  **PADDING)
        p.min_label.grid(column=4, row=row, **PADDING)
        p.min_entry.grid(column=5, row=row, **PADDING)
        p.max_label.grid(column=6, row=row, **PADDING)
        p.max_entry.grid(column=7, row=row, **PADDING)
        p.fixed_label.grid(column=8, row=row, **PADDING)
        p.fixed_entry.grid(column=9, row=row, **PADDING)
    
    def checkbox_event(self, p: ModelParameter) -> None:
        if p.vary.get():
            p.fixed_entry.configure(state="disabled", fg_color=DISABLE_COLOR)
            p.min_entry.configure(state="normal", fg_color=ENABLE_COLOR)
            p.max_entry.configure(state="normal", fg_color=ENABLE_COLOR)
        else:
            p.fixed_entry.configure(state="normal", fg_color=ENABLE_COLOR)
            p.min_entry.configure(state="disabled", fg_color=DISABLE_COLOR)
            p.max_entry.configure(state="disabled", fg_color=DISABLE_COLOR)

class ToplevelWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__()
        self.master = master

        canvas = FigureCanvasTkAgg(master=self, figure=self.master.fig) # Convert the Figure to a tkinter widget
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
        self.rng_key = random.PRNGKey(0)

        intro_frame = ctk.CTkFrame(self)
        intro_frame.pack()
        ctk.CTkLabel(intro_frame, text="Welcome!\nFill required information in all tabs\nthen click 'Submit' and 'Fit Spectra'.\nClick 'Show Fit' to view plot of fit.").grid(row=0, column=0, **PADDING, columnspan=2)
        appearance_mode_label = ctk.CTkLabel(intro_frame, text="Appearance Mode:", anchor="w")
        appearance_mode_label.grid(row=1, column=0, **PADDING)
        appearance_mode_optionemenu = ctk.CTkOptionMenu(intro_frame, values=["Light", "Dark"], command=self.change_appearance_mode_event)
        appearance_mode_optionemenu.grid(row=1, column=1, **PADDING)
        appearance_mode_optionemenu.set("Light")
        
        self.tab_view = MyTabView(self)
        self.tab_view.pack(fill="both", expand=True)
        buttons_frame = ctk.CTkFrame(self)

        buttons_frame.pack(anchor="center", fill="both", expand=True, pady=10)
        ctk.CTkButton(buttons_frame, width=75, text="Submit", command=self.sumbit_entries).pack(side="left", padx=10, expand=True)
        self.perform_fit_btn = ctk.CTkButton(buttons_frame, width=75, text="Fit Spectra", command=self.fit_spectra, state="disabled")
        self.perform_fit_btn.pack(side="left", padx=10, expand=True)
        self.show_fit_btn = ctk.CTkButton(buttons_frame, width=75, text="Show Fit", command=self.show_fit, state="disabled")
        self.show_fit_btn.pack(side="left", padx=10, expand=True)
        self.save_fit_btn = ctk.CTkButton(buttons_frame, width=75, text="Save Fit", command=self.save_fit, state="disabled")
        self.save_fit_btn.pack(side="left", padx=10, expand=True)
        
        # flag to make sure that results windows is not displayed twice
        self.toplevel_window = None

    def sumbit_entries(self) -> None:
        ct = self.tab_view.tab("Set Constants") # to access attributes of constants tab
        ft = self.tab_view.tab("Set Fit Parameters") # to access attributes of fit pars tab
        dt = self.tab_view.tab("Select Data") # to access attributes of data selection tab
        try:
            for p in [ct.B0, ct.gamma, ct.tp, ft.R1a, ft.R2a, ft.dwa, ft.R1b, ft.R2b, ft.k, ft.f, ft.dwb]:
                p.get_entries()
            for p in [ft.R1a, ft.R2a, ft.dwa, ft.R1b, ft.R2b, ft.k, ft.f, ft.dwb]:
                p.set_prior()

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
            CTkMessagebox(title="Info", message="Entries submitted successfully!\nClick 'Fit Spectra' to proceed.",
                        icon="check", wraplength=300)
            self.perform_fit_btn.configure(state="normal")
        except:
            CTkMessagebox(title="Error", message="Please fill all required fields\nand select data file.", icon="warning", wraplength=300)

    def fit_spectra(self) -> None:
        def model(self):
            model_pars = jnp.array([
                numpyro.sample(p.name, p.prior) if p.vary.get() else p.fixed_value for p in [self.R1a, self.R2a, self.dwa, self.R1b, self.R2b, self.k, self.f, self.dwb]
            ])
            sigma = numpyro.sample("sigma", dist.HalfNormal(0.03))
            model_pred = bloch_mcconnell(model_pars, self.offsets, self.powers, self.B0, self.gamma, self.tp)
            numpyro.sample("obs", dist.Normal(model_pred, sigma), obs=self.data)
        
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(model, init_strategy=numpyro.infer.init_to_mean),
            num_warmup=1000,
            num_samples=2000,
            num_chains=2,
            chain_method="sequential",
            progress_bar=True
        )
        mcmc.run(self.rng_key, self, extra_fields=('potential_energy', 'energy'))
        self.idata = az.from_numpyro(posterior=mcmc)
        self.fit_summary = az.summary(self.idata, round_to=5, stat_funcs={'median': np.median, 'mode': lambda x: az.plots.plot_utils.calculate_point_estimate('mode', x)}, var_names=["~sigma"])

        print(self.fit_summary.index.to_list())

        """ TO DO
        # ADD FAKE PROGRESS BAR FOR SAMPLING DURATION
        # HAVE AN OPTION MENU NEAR 'SHOW RESULTS' BUTTON TO PICK METHOD OF SUMMARIZING POSTERIOR: MEAN, MEDIAN, MODE.
        # 'SHOW RESULTS' BUTTON SHOULD OPEN A TOP-LEVEL WINDOW THAT CONTAINS:
            LEFT: FIT SUMMARY; MENTION CONSTANTS
            RIGHT: PLOT OF FIT; BY SELECTED METHOD
            BOTTOM: SERIES OF BUTTONS THAT OPEN IN OTHER TOP-LEVEL WINDOWS:
                - PAIR PLOT
                - ENERGY PLOT
                - TRACE PLOT
                - ESS PLOT
        """

        # self.best_fit_pars = np.asarray(
        #     [self.fit.params[f"{p.name}"].value for p in [self.R1a, self.R2a, self.dwa, self.R1b, self.R2b, self.k, self.f, self.dwb]]
        #     )
        # self.best_fit_spectra = bloch_mcconnell(self.best_fit_pars, self.offsets, self.powers, self.B0, self.gamma, self.tp)

        CTkMessagebox(self, title="Info", message="Done!", icon="check")
        self.show_fit_btn.configure(state="normal")
        self.save_fit_btn.configure(state="normal")

        # self.fig, ax = plt.subplots()
        # ax.plot(self.offsets, self.best_fit_spectra.T)
        # ax.set_prop_cycle(None)
        # ax.plot(self.offsets, self.data.T, '.', label=[f"{power:.1f} μT" for power in self.powers])
        # ax.set_xlabel("offset [ppm]")
        # ax.set_ylabel("Z-value [a.u.]")
        # ax.set_title("Nonlinear Least Squares Fit")
        # ax.legend()
        # plt.close(self.fig)

    def show_fit(self):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = ToplevelWindow(self)  # create window if its None or destroyed
        else:
            self.toplevel_window.focus()  # if window exists focus it
        
    def save_fit(self):
        savedir = filedialog.askdirectory(title="Select Save Directory")
        with open(os.path.join(savedir,os.path.join(savedir,"fit.txt")), "w") as text_file:
            text_file.write(lmfit.printfuncs.fit_report( self.fit))

        df = pd.DataFrame(np.c_[self.offsets, self.best_fit_spectra.T], columns=["ppm"] + [f"{power:.1f} μT" for power in self.powers])
        with pd.ExcelWriter(os.path.join(savedir,"fit.xlsx")) as writer:
            df.to_excel(writer, index=False)

        self.fig.savefig(os.path.join(savedir,"fit.pdf"))

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)


@partial(jnp.vectorize, excluded=[0,1,3,4,5], signature="()->(k)") # powers
@partial(jnp.vectorize, excluded=[0,2,3,4,5], signature="()->()") # offsets
def bloch_mcconnell(model_pars, offset: float, power: float, B0:float, gamma:float, tp:float):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_pars
    return Z_analytical_symbolic(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp)




app = App()
app.mainloop()
