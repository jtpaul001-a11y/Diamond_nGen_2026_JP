# building the MIT-VaultLab neutron generator
# angular and energy distribution

from pathlib import Path
from collections.abc import Iterable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sum_overlapping_spectra(
    energy_1,
    spectrum_1,
    energy_2,
    spectrum_2,
):

    """
    Sums two overlapping energy spectra while checking that the energy bins match.
    Thanks Claude!
    """

    energy_1 = np.asarray(energy_1)
    spectrum_1 = np.asarray(spectrum_1)

    energy_2 = np.asarray(energy_2)
    spectrum_2 = np.asarray(spectrum_2)

    if len(energy_1) != len(spectrum_1):
        raise ValueError("energy_1 and spectrum_1 must have matching lengths.")

    if len(energy_2) != len(spectrum_2):
        raise ValueError("energy_2 and spectrum_2 must have matching lengths.")

    # Find the common energy range
    lower_limit = max(energy_1[0], energy_2[0])
    upper_limit = min(energy_1[-1], energy_2[-1])

    if lower_limit > upper_limit:
        raise ValueError("The energy arrays do not overlap.")

    # Select only bins inside the shared range
    mask_1 = (energy_1 >= lower_limit) & (energy_1 <= upper_limit)
    mask_2 = (energy_2 >= lower_limit) & (energy_2 <= upper_limit)

    common_energy_1 = energy_1[mask_1]
    common_energy_2 = energy_2[mask_2]

    # Verify that the overlapping bins correspond
    if not np.array_equal(np.round(common_energy_1, 6), np.round(common_energy_2, 6)):
        raise ValueError("The overlapping bins do not have matching binning.")

    avg_spectrum = (
        spectrum_1[mask_1] +
        spectrum_2[mask_2]
    ) / 2

    return common_energy_1, avg_spectrum


def nGen_diamond(N_char = 1, center=(0, 0, 0),
                  reference_uvw=(0, 0, 1), Show_plots=False
                  ) -> "Iterable[IndependentSource]":
    """
    Builds the MIT-VaultLab nGen-400 neutron generator in OpenMC
    with data tabulated from Jon Paul and Collin Dunn characterization
    via diamond detectors

    The spectra files are pickled and stored as spectra[name]['average_flux']
    and spectra[name]['energies'], which return respectively the time-averaged
    flux spectra and the corresponding energy bins. The name indicates the angle,
    for the first characterization the angles are integers from -135 to +135
    written as '-135deg', for example, and for the second characterization 
    additional angles were taken just about the beam axis and so one decimal 
    is used, for example '-135.0deg'... '5.4deg'...

    N_char : int, optional
        1 uses the data from the first characterization done in february 2026, 
          data shows more simmetry
        2 uses the data from the second characterization done in April 2026, 
          data shows more asymmetry
    In both cases, the data is averaged over pos and neg angles such that
    symmetry is forced

    Show_plots : bool, optional
        If True, plots of the spectra will be shown, by default False

    Uses the pickled data file spectra.pkl, which contains the spectra
    for the nGen-400 neutron generator, across a range of polar angles

    Parameters
    ----------
    center : tuple, optional
        coordinate position of the source (it is a point source),
        by default (0, 0, 0)
    reference_uvw : tuple, optional
        direction for the polar angle (tuple or list of versors)
    it is the same for the openmc.PolarAzimuthal class
    more specifically, polar angle = 0 is the direction of the D accelerator
    towards the Ti-T target, by default (0, 0, 1)

    Returns
    -------
        list of openmc neutron sources with angular and energy distribution
        and total strength of 1
    """
    try:
        import pickle
        import openmc
    except ModuleNotFoundError:
        raise ModuleNotFoundError("openmc and pickle are required")

    if N_char == 1:
        filename = "spectra.pkl"
    else:
        filename = "spectra2.pkl"

    filename = str(Path(__file__).parent) / Path(filename)
    print("filename:", filename)

    with Path(filename).open("rb") as source:
        spectra = pickle.load(source)
        
    # Take the angles and create an average from neg to pos angles, to force symmetry
    sym_spectra = {}
    sym_angles = []

    for name in list(spectra.keys()):
        angle = float(name.removesuffix("deg"))

        if angle < -1.0:
            pos_name = f"{-angle:.0f}deg" if N_char == 1 else f"{-angle:.1f}deg"
            sym_spectra[pos_name] = {}

            if pos_name in list(spectra.keys()):
                sym_angles.append(-angle)
                energies, avg_spectrum = sum_overlapping_spectra(
                    spectra[pos_name]['energies'],
                    np.clip(spectra[pos_name]['average_flux'], 0, np.inf),
                    spectra[name]['energies'],
                    np.clip(spectra[name]['average_flux'], 0, np.inf),
                )
                sym_spectra[pos_name]['energies'] = energies
                sym_spectra[pos_name]['average_flux'] = avg_spectrum

            else:
                raise ValueError(f"Missing positive angle for {name}, searching for {pos_name} in spectra keys: {list(spectra.keys())}")

        elif angle == 0.0:
            sym_spectra[name] = {}
            sym_spectra[name]['energies'] = spectra[name]['energies']
            sym_spectra[name]['average_flux'] = np.clip(spectra[name]['average_flux'], 0, np.inf)
            sym_angles.append(angle)
        else:
            continue

        if Show_plots:
            fig, ax = plt.subplots()
            ax.plot(energies, avg_spectrum, label=f"avg {pos_name}") if angle != 0.0 else None
            ax.plot(spectra[pos_name]['energies'], np.clip(spectra[pos_name]['average_flux'], 0, np.inf), linestyle="--", label=f"pos {pos_name}", alpha=0.5) if angle != 0.0 else None
            ax.plot(spectra[name]['energies'], np.clip(spectra[name]['average_flux'], 0, np.inf), linestyle="-", label=f"{name}", alpha=0.5)
            ax.set_xlabel("Energy (MeV)")
            ax.set_ylabel("Average Flux (Arb. Units)")
            ax.set_title("Symmetric Spectra")
            plt.grid()
            plt.legend()
    plt.show() if Show_plots else None

    # Sort the angles and get the corresponding yields
    sym_angles = np.sort(sym_angles)
    diff_arr = np.diff(sym_angles)
    yields = np.zeros(len(sym_angles))

    for i, sym_angle in enumerate(sym_angles):
        name = f"{sym_angle:.0f}deg" if N_char == 1 else f"{sym_angle:.1f}deg"
        yields[i] = sym_spectra[name]['average_flux'].sum()

    yields /= np.sum(yields) # Normalize source strengths to 1

    # azimuthal values
    phi = openmc.stats.Uniform(a=0, b=2 * np.pi)
    
    all_sources = []
    for i, angle in enumerate(sym_angles):
        name = f"{sym_angles[i]:.0f}deg" if N_char == 1 else f"{sym_angles[i]:.1f}deg"
        if i == 0:
            # Start at zero and go halfway to the next angle
            mu = openmc.stats.Uniform(a=np.cos(np.deg2rad(angle)), b=np.cos(np.deg2rad(angle+diff_arr[i]/2)))  # Go halfway to the next angle
        elif i == len(sym_angles) - 1:
            # Go evenly past the last angle with respect to the previous angle
            mu = openmc.stats.Uniform(a=np.cos(np.deg2rad(angle-diff_arr[i-1]/2)), b=np.cos(np.deg2rad(angle+diff_arr[i-1]/2)))  # Go halfway to the previous angle
        else:
            # Go halfway to the previous and next angle
            mu = openmc.stats.Uniform(a=np.cos(np.deg2rad(angle-diff_arr[i-1]/2)), 
                                      b=np.cos(np.deg2rad(angle+diff_arr[i]/2)))  
        #print(f"i: {i}, angle: {angle}, mu: {mu.a:.4f} to {mu.b:.4f}")

        space = openmc.stats.Point(center)
        angle = openmc.stats.PolarAzimuthal(mu=mu, phi=phi, reference_uvw=reference_uvw)
        energy = openmc.stats.Tabular(
            sym_spectra[name]['energies'], sym_spectra[name]['average_flux'],
              interpolation="linear-linear"
        )
        strength = yields[i]
        my_source = openmc.IndependentSource(
            space=space,
            angle=angle,
            energy=energy,
            strength=strength,
            particle="neutron",
        )
        all_sources.append(my_source)
    return all_sources


#source = nGen_diamond(N_char=1, center=(0, 0, 0), reference_uvw=(0, 0, 1), Show_plots=True)