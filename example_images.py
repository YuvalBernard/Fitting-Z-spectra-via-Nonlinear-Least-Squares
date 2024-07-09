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
from threading import Thread


# Configurations
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

# Constants
ENABLE_COLOR = "#F9F9FA"
DISABLE_COLOR = "gray30"
ENTRY_WIDTH = 130
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
            self.label = customtkinter.CTkLabel(master, text=f"{self.name} [{self.units}]", anchor="w")
            self.entry = customtkinter.CTkEntry(master, width=ENTRY_WIDTH, corner_radius=0)

    def get_entries(self) -> None:

        def get_entry(entry: str):
            if len(split_entries := entry.split(',')) > 1:
                return np.asarray(split_entries, dtype=float)
            else:
                return float(entry)

        if self.vary is not None:
            par_varies = self.vary.get()
            self.min = float(self.min_entry.get()) if par_varies else None
            self.max = float(self.max_entry.get()) if par_varies else None
            self.fixed_value = None if par_varies else float(self.fixed_entry.get())
        else:
            self.value = get_entry(self.entry.get())

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
        self.configure(fg_color="transparent")

        # Data
        customtkinter.CTkButton(self, text="Browse Data File", command=self.browsefunc).grid(row=0, column=0, **PADDING, sticky="w")
        self.file_label = customtkinter.CTkLabel(self, text="No file selected.", anchor="w")
        self.file_label.grid(row=0, column=1, **PADDING, sticky="w")

        # Constants
        self.B0 = ModelParameter(name="B₀", units="T", description="Static field strength")
        self.gamma = ModelParameter(name="γ", units="10⁶ rad⋅s⁻¹⋅T⁻¹", description="Gyromagnetic ratio")
        self.tp = ModelParameter(name="tₚ", units="s", description="Saturation pulse duration")
        self.powers = ModelParameter(name="ω₁ (list)", units="μT", description="Irradiation amplitudes (comma separated list)")

        for row, p in enumerate([self.B0, self.gamma, self.tp, self.powers]):
            self.create_fit_par_widgets(p, row+1)

        # for testing purposes. remove in main version
        self.B0.entry.insert(0, 7.4)
        self.gamma.entry.insert(0, 103.962)
        self.tp.entry.insert(0, 2.0)

        # Example table
        customtkinter.CTkLabel(self, text="Example Table: (Header must have specific form)", anchor="w").grid(row=1, column=2, padx=50, pady=PADDING["pady"], sticky="w")
        table_image = customtkinter.CTkImage(Image.open("example_data.png"), size=(250, 200))
        customtkinter.CTkLabel(self, image=table_image, text="").grid(row=2, column=2, rowspan=4, padx=50, pady=5, sticky="w")

    def browsefunc(self) -> None:
        filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if filename:
            df = pd.read_excel(filename)
            powers = list(df.columns[1:].str.extract(r"(\d+.\d+)", expand=False))
            self.powers.entry.delete(0, customtkinter.END)
            self.powers.entry.insert(0, ", ".join(powers))
            self.offsets = df["ppm"].to_numpy(dtype=float)
            self.data = df.to_numpy(dtype=float).T[1:]
            self.file_label.configure(text=f"Selected file: {os.path.relpath(path=filename, start=os.getcwd())}", font=customtkinter.CTkFont(underline=True))
    
    def create_fit_par_widgets(self, p: ModelParameter, row: int) -> None:
        p.set_entries_and_labels(self)
        p.label.grid(column=0, row=row, **PADDING, sticky="w")
        CTkToolTip(p.label, message=p.description, alpha=0.9)
        p.entry.grid(column=1, row=row, **PADDING, sticky="w")

class FitParsFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.configure(fg_color="transparent")

        customtkinter.CTkLabel(self, text="Tip: Hover over parameter label for a short description", anchor="w").pack(**PADDING)

        self.R1a = ModelParameter(name="R1a", units="Hz", vary=False, description="Longitudinal relaxation rate of pool a")
        self.R2a = ModelParameter(name="R2a", units="Hz", vary=False, description="Transverse relaxation rate of pool a")
        self.dwa = ModelParameter(name="Δωa", units="ppm", vary=False, description="Larmor frequency of pool a relative to itself.\nShould be zero")
        self.R1b = ModelParameter(name="R1b", units="Hz", vary=True, description="Longitudinal relaxation rate of pool b")
        self.R2b = ModelParameter(name="R2b", units="Hz", vary=True, description="Transverse relaxation rate of pool b")
        self.k = ModelParameter(name="k", units="Hz", vary=True, description="Exchange rate from pool b to pool a")
        self.f = ModelParameter(name="f", units=None, vary=True, description=("Equilibrium magnetization of pool b relative to pool a."
                                                                         "\nRoughly equivalent to fraction of pool b in solution"))
        self.dwb = ModelParameter(name="Δωb", units="ppm", vary=True, description="Larmor frequency of pool b relative to pool a")

        self.pars_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.pars_frame.pack(**PADDING)
        for row, p in enumerate([self.R1a, self.R2a, self.dwa, self.R1b, self.R2b, self.k, self.f, self.dwb]):
            self.create_fit_par_widgets(p, row)

        # for testing purposes. remove in main version
        self.R1a.fixed_entry.insert(0, 8.0)
        self.R2a.fixed_entry.insert(0, 380)
        self.dwa.fixed_entry.insert(0, 0)
        self.R1b.min_entry.insert(0, 0.1)
        self.R1b.max_entry.insert(0, 100.0)
        self.R2b.min_entry.insert(0, 1000)
        self.R2b.max_entry.insert(0, 100_000)
        self.k.min_entry.insert(0, 1)
        self.k.max_entry.insert(0, 500)
        self.f.min_entry.insert(0, 1e-5)
        self.f.max_entry.insert(0, 0.1)
        self.dwb.min_entry.insert(0, -265)
        self.dwb.max_entry.insert(0, -255)

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

class ProgressBarWindow(customtkinter.CTkToplevel):
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.title("MCMC Sampling Progress Tracker")

        self.mcmc_status = customtkinter.StringVar(value="Running MCMC. Please wait...")
        self.mcmc_status_label = customtkinter.CTkLabel(self, textvariable=self.mcmc_status)
        self.mcmc_status_label.pack(**PADDING)
        progressbar = customtkinter.CTkProgressBar(self, mode="indeterminate")
        progressbar.pack(**PADDING)
        progressbar.start()

        def run_mcmc(self, master):
            master.mcmc.run(master.rng_key, master, extra_fields=('potential_energy', 'energy'))
            master.idata = az.from_numpyro(posterior=master.mcmc)
            master.fit_summary = az.summary(master.idata, round_to=5, stat_funcs={'median': np.median, 'mode': lambda x: az.plots.plot_utils.calculate_point_estimate('mode', x)}, var_names=["~sigma"])
            self.mcmc_status.set("Finished Sampling.")

        Thread(target=run_mcmc, args=(self, self.master)).start()
        self.mcmc_status_label.wait_variable(self.mcmc_status)
        progressbar.stop()
        self.destroy()

class ResultsFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.configure(fg_color="transparent")
        self.master = master

        # Add a summary table somehow.
        self.posterior_summary_menu = customtkinter.CTkOptionMenu(self, values=["mean", "median", "mode"],
                                                                command=self.change_posterior_summary_event)
        self.posterior_summary_menu.grid(row=0, column=0, **PADDING)

        def get_fit_plot(master, method: str):
            master.fig, ax = plt.subplots()
            match method:
                case "mean":
                    ax.plot(master.offsets, master.best_fit_spectra_mean.T)
                case "median":
                    ax.plot(master.offsets, master.best_fit_spectra_median.T)
                case "mode":
                    ax.plot(master.offsets, master.best_fit_pars_mode.T)
            ax.set_prop_cycle(None)
            ax.plot(master.offsets, master.data.T, '.', label=[f"{power:.1f} μT" for power in master.powers])
            ax.set_xlabel("offset [ppm]")
            ax.set_ylabel("Z-value [a.u.]")
            ax.set_title("Nonlinear Least Squares Fit")
            ax.legend()
            plt.close(master.fig)
            return master.fig

        self.canvas = FigureCanvasTkAgg(master=self, figure=get_fit_plot(self.master, "mean")) # Convert the Figure to a tkinter widget
        self.canvas.draw() # Draw the graph on the canvas
        self.canvas.get_tk_widget().grid(row=0, column=1, **PADDING) # Show the widget on the screen

    def change_posterior_summary_event(self, method):
        self.canvas.configure(figure=get_fit_plot(self.master, method))


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.rng_key = random.PRNGKey(0)

        self.title("image_example.py")
        self.geometry("1350x750")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_images")
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "weizmann_logo.png")), size=(200, 40))
        self.data_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "excel_icon_light.png")), size=(20, 20))
        self.pars_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "lambda_icon_light.png")), size=(20, 20))
        self.results_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "result_icon_light.png")), size=(20, 20))
        self.submit_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "submit_icon.png")), size=(20, 20))
        self.perform_fit_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "magnifying_glass_icon.png")), size=(20,20))

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
                                                      image=self.results_image, anchor="w", command=self.results_button_event, state="disabled")
        self.results_button.grid(row=3, column=0, sticky="ew")

        self.submit_entries_button = customtkinter.CTkButton(self.navigation_frame, text="Submit Entries",
                                                                image=self.submit_image, command=self.submit_entries)
        self.submit_entries_button.grid(row=5, column=0, **PADDING, sticky="s")

        self.perform_fit_button = customtkinter.CTkButton(self.navigation_frame, text="Perform Fit",
                                                                image=self.perform_fit_image, command=self.perform_fit, state="disabled")
        self.perform_fit_button.grid(row=6, column=0, **PADDING, sticky="s")

        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=7, column=0, **PADDING, sticky="s")

        # create data and constants frame
        self.data_constants_frame = DataConstantsFrame(self)
        self.data_constants_frame.grid()

        # create model parameters frame
        self.pars_frame = FitParsFrame(self)
        self.pars_frame.grid()

        self.results_frame = customtkinter.CTkFrame(self)

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
    
    def submit_entries(self):
        cdt = self.data_constants_frame # to access attributes of constants and data tab
        fpt = self.pars_frame # to access attributes of fit pars tab
        try:
            for p in [cdt.B0, cdt.gamma, cdt.tp, cdt.powers, fpt.R1a, fpt.R2a, fpt.dwa, fpt.R1b, fpt.R2b, fpt.k, fpt.f, fpt.dwb]:
                p.get_entries()
            for p in [fpt.R1a, fpt.R2a, fpt.dwa, fpt.R1b, fpt.R2b, fpt.k, fpt.f, fpt.dwb]:
                p.set_prior()

            self.B0 = cdt.B0.value
            self.gamma = cdt.gamma.value
            self.tp = cdt.tp.value
            self.powers = cdt.powers.value
            self.offsets = cdt.offsets
            self.data = cdt.data
            self.R1a = fpt.R1a
            self.R2a = fpt.R2a
            self.dwa = fpt.dwa
            self.R1b = fpt.R1b
            self.R2b = fpt.R2b
            self.k = fpt.k
            self.f = fpt.f
            self.dwb = fpt.dwb

            CTkMessagebox(title="Info", message="Entries submitted successfully!\nClick 'Perform fit' to proceed.",
                        icon="check", wraplength=300)
            self.perform_fit_button.configure(state="normal")

        except:
            CTkMessagebox(title="Error", message="Please fill all required fields\nand select data file.", icon="warning", wraplength=300)

    
    def perform_fit(self):
        def model(self):
            model_pars = jnp.asarray([
                numpyro.sample(p.name, p.prior) if p.vary.get() else p.fixed_value for p in [self.R1a, self.R2a, self.dwa, self.R1b, self.R2b, self.k, self.f, self.dwb]
            ])
            sigma = numpyro.sample("sigma", dist.HalfNormal(0.03))
            model_pred = bloch_mcconnell(model_pars, self.offsets, self.powers, self.B0, self.gamma, self.tp)
            numpyro.sample("obs", dist.Normal(model_pred, sigma), obs=self.data)
        
        self.mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(model, init_strategy=numpyro.infer.init_to_mean),
            num_warmup=100, # should be 1000
            num_samples=200, # should be 200
            num_chains=2, # should be 4
            chain_method="sequential",
            progress_bar=True
        )

        # Open progress bar in top-level window
        self.toplevel_window = None # flag to make sure that results windows is not displayed twice
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = ProgressBarWindow(self)  # create window if its None or destroyed
        else:
            self.toplevel_window.focus()  # if window exists focus it
        print(self.fit_summary)
        self.best_fit_pars_mean = np.asarray([
            self.fit_summary["mean"][p.name] if p.vary.get() else p.fixed_value for p in [self.R1a, self.R2a, self.dwa, self.R1b, self.R2b, self.k, self.f, self.dwb]
        ])
        self.best_fit_spectra_mean = bloch_mcconnell(self.best_fit_pars_mean, self.offsets, self.powers, self.B0, self.gamma, self.tp)
        self.best_fit_pars_median = np.asarray([
            self.fit_summary["median"][p.name] if p.vary.get() else p.fixed_value for p in [self.R1a, self.R2a, self.dwa, self.R1b, self.R2b, self.k, self.f, self.dwb]
        ])
        self.best_fit_spectra_median = bloch_mcconnell(self.best_fit_pars_median, self.offsets, self.powers, self.B0, self.gamma, self.tp)
        self.best_fit_pars_mode = np.asarray([
            self.fit_summary["mode"][p.name] if p.vary.get() else p.fixed_value for p in [self.R1a, self.R2a, self.dwa, self.R1b, self.R2b, self.k, self.f, self.dwb]
        ])
        self.best_fit_spectra_mode = bloch_mcconnell(self.best_fit_pars_mode, self.offsets, self.powers, self.B0, self.gamma, self.tp)

        self.results_button.configure(state="normal")
        CTkMessagebox(self, title="Info", message="Done!", icon="check", wraplength=300)

        self.results_frame = ResultsFrame(self)
        self.results_frame.grid()



@partial(jnp.vectorize, excluded=[0,1,3,4,5], signature="()->(k)") # powers
@partial(jnp.vectorize, excluded=[0,2,3,4,5], signature="()->()") # offsets
def bloch_mcconnell(model_pars, offset, power, B0, gamma, tp):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_pars
    return Z_analytical_symbolic(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp)


if __name__ == "__main__":
    app = App()
    app.mainloop()
