import customtkinter
from customtkinter import filedialog
from CTkMessagebox import CTkMessagebox
from CTkToolTip import CTkToolTip
from PIL import Image
from typing import Optional
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import numpy as np
import jax.numpy as jnp
from solve_bloch_mcconnell import Z_analytical_symbolic
from jax import jit, config, random
from jax.scipy.stats import norm
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
import arviz as az
import pandas as pd


# Constants
ENABLE_COLOR = "#F9F9FA"
DISABLE_COLOR = "gray40"
ENTRY_WIDTH = 100
PADDING = {'padx':20, 'pady':20}

customtkinter.set_appearance_mode("light")

# Helper classes
class ModelParameter:
    def __init__(self, name: str, units: str | None, description: str, vary: Optional[bool]=None):
        self.name = name
        self.units = units
        self.vary = vary
        self.description = description

    def set_entries_and_labels(self, master: customtkinter.CTkFrame) -> None:
        if self.vary is not None:
            if self.units is None:
                self.label = customtkinter.CTkLabel(master, text=f"{self.name}")
            else:
                self.label = customtkinter.CTkLabel(master, text=f"{self.name} [{self.units}]")
            self.min_label = customtkinter.CTkLabel(master, text="min", anchor="w")
            self.max_label = customtkinter.CTkLabel(master, text="max", anchor="w")
            self.fixed_label = customtkinter.CTkLabel(master, text="fixed value", anchor="w")
            self.min_entry = customtkinter.CTkEntry(master, width=ENTRY_WIDTH, corner_radius=0, state="normal" if self.vary else "disabled", fg_color=ENABLE_COLOR if self.vary else DISABLE_COLOR)
            self.max_entry = customtkinter.CTkEntry(master, width=ENTRY_WIDTH, corner_radius=0, state="normal" if self.vary else "disabled", fg_color=ENABLE_COLOR if self.vary else DISABLE_COLOR)
            self.fixed_entry = customtkinter.CTkEntry(master, width=ENTRY_WIDTH, corner_radius=0, state="disabled" if self.vary else "normal", fg_color=DISABLE_COLOR if self.vary else ENABLE_COLOR)
        else:
            self.label = customtkinter.CTkLabel(master, text=f"{self.name} [{self.units}]")
            self.entry = customtkinter.CTkEntry(master, width=ENTRY_WIDTH, corner_radius=0)

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


class DataConstantsFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.configure(fg_color="transparent")

        # Data
        customtkinter.CTkButton(self, text="Browse data file", command=self.browsefunc).grid(row=0, column=0, **PADDING, sticky="w")
        self.file_label = customtkinter.CTkLabel(self, text="No file selected.", anchor="w")
        self.file_label.grid(row=0, column=1, **PADDING, sticky="w")

        # Constants
        # customtkinter.CTkLabel(self, text="Tip: Hover over parameter label for a short description").pack(**PADDING)
        
        self.master.B0 = ModelParameter(name="B₀", units="T", description="Static field strength")
        self.master.gamma = ModelParameter(name="γ", units="10⁶ rad⋅s⁻¹⋅T⁻¹", description="Gyromagnetic ratio")
        self.master.tp = ModelParameter(name="tₚ", units="s", description="Saturation pulse duration")
        self.master.powers = ModelParameter(name="ω₁", units="μT", description="Irradiation amplitudes (list)")

        for row, p in enumerate([self.master.B0, self.master.gamma, self.master.tp, self.master.powers]):
            self.create_fit_par_widgets(p, row, column=3)
    
    def browsefunc(self) -> None:
        filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if filename:
            self.df = pd.read_excel(filename)
            powers = list(self.df.columns[1:].str.extract(r"(\d+.\d+)", expand=False))
            self.master.offsets = np.asarray(self.df["ppm"].to_numpy().astype(float))
            self.master.powers = np.asarray(powers, dtype=float)
            self.master.data = np.asarray(self.df.to_numpy().astype(float).T[1:])
            self.file_label.configure(text=f"Selected file: {os.path.relpath(path=filename, start=os.getcwd())}")
    
    def create_fit_par_widgets(self, p: ModelParameter, row: int, column: int) -> None:
        p.set_entries_and_labels(self)
        p.label.grid(column=column, row=row, **PADDING)
        CTkToolTip(p.label, message=p.description, alpha=0.9)
        p.entry.grid(column=column+1, row=row, **PADDING)


class ConstantsFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        customtkinter.CTkLabel(self, text="Tip: Hover over parameter label for a short description").pack(**PADDING)
        
        self.master.B0 = ModelParameter(name="B₀", units="T", description="Static field strength")
        self.master.gamma = ModelParameter(name="γ", units="10⁶ rad⋅s⁻¹⋅T⁻¹", description="Gyromagnetic ratio")
        self.master.tp = ModelParameter(name="tₚ", units="s", description="Saturation pulse duration")

        self.pars_frame = customtkinter.CTkFrame(self)
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


class FitParsFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.configure(fg_color="transparent")

        customtkinter.CTkLabel(self, text="Tip: Hover over parameter label for a short description", anchor="w").pack(**PADDING)

        self.master.R1a = ModelParameter(name="R1a", units="Hz", vary=False, description="Longitudinal relaxation rate of pool a")
        self.master.R2a = ModelParameter(name="R2a", units="Hz", vary=False, description="Transverse relaxation rate of pool a")
        self.master.dwa = ModelParameter(name="Δωa", units="ppm", vary=False, description="Larmor frequency of pool a relative to itself.\nShould be zero")
        self.master.R1b = ModelParameter(name="R1b", units="Hz", vary=True, description="Longitudinal relaxation rate of pool b")
        self.master.R2b = ModelParameter(name="R2b", units="Hz", vary=True, description="Transverse relaxation rate of pool b")
        self.master.k = ModelParameter(name="k", units="Hz", vary=True, description="Exchange rate from pool b to pool a")
        self.master.f = ModelParameter(name="f", units=None, vary=True, description=("Equilibrium magnetization of pool b relative to pool a."
                                                                         "\nRoughly equivalent to fraction of pool b in solution"))
        self.master.dwb = ModelParameter(name="Δωb", units="ppm", vary=True, description="Larmor frequency of pool b relative to pool a")

        self.pars_frame = customtkinter.CTkFrame(self, fg_color="transparent")
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
        p.vary = customtkinter.BooleanVar(value=p.vary)
        customtkinter.CTkCheckBox(self.pars_frame, text="vary", variable=p.vary, command=lambda : self.checkbox_event(p)).grid(column=1, row=row,  **PADDING)
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

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.rng_key = random.PRNGKey(0)

        self.title("image_example.py")
        self.geometry("1320x600")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_images")
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "weizmann_logo.png")), size=(200, 40))
        self.data_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "excel_icon_light.png")), size=(20, 20))
        self.pars_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "lambda_icon_light.png")), size=(20, 20))
        self.results_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "result_icon_light.png")), size=(20, 20))

        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="", image=self.logo_image,
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.data_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Set Data and Constants",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                   image=self.data_image, anchor="w", command=self.data_button_event)
        self.data_button.grid(row=1, column=0, sticky="ew")

        self.pars_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Set Model Parameters",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      image=self.pars_image, anchor="w", command=self.pars_button_event)
        self.pars_button.grid(row=2, column=0, sticky="ew")

        self.results_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Show and Save Results",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      image=self.results_image, anchor="w", command=self.results_button_event)
        self.results_button.grid(row=3, column=0, sticky="ew")

        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=20, sticky="s")

        # create data and constants frame
        self.data_constants_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        # data frame
        self.data_frame = DataConstantsFrame(self.data_constants_frame)
        self.data_frame.grid()

        # create model parameters frame
        self.pars_frame = FitParsFrame(self)
        self.pars_frame.grid()
        # create results frame
        self.results_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        # select default frame
        self.select_frame_by_name("data")

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.data_button.configure(fg_color=("gray75", "gray25") if name == "data" else "transparent")
        self.pars_button.configure(fg_color=("gray75", "gray25") if name == "pars" else "transparent")
        self.results_button.configure(fg_color=("gray75", "gray25") if name == "results" else "transparent")

        # show selected frame
        if name == "data":
            self.data_constants_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.data_constants_frame.grid_forget()
        if name == "pars":
            self.pars_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.pars_frame.grid_forget()
        if name == "results":
            self.results_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.results_frame.grid_forget()

    def data_button_event(self):
        self.select_frame_by_name("data")

    def pars_button_event(self):
        self.select_frame_by_name("pars")

    def results_button_event(self):
        self.select_frame_by_name("results")

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)


def interpret_entry(entry: str):
    if len(split_entries := entry.split(',')) > 1:
        return [float(i) for i in split_entries]
    else:
        return float(entry)

if __name__ == "__main__":
    app = App()
    app.mainloop()
