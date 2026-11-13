
import openmc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

N_source_particles = 1000000
N_batches = 5

filename = f"results_{N_source_particles}_particles_{N_batches}_batches"
filename = str(Path(__file__).parent) / Path(filename)
print("filename:", filename)

run_dir = filename
sp = openmc.StatePoint(f'{run_dir}/statepoint.{N_batches}.h5')
angles = np.arange(-135, 135.1, 9)
energies = np.arange(13.0e6, 15.6e6, 2e4) # eV

cmap = plt.get_cmap('viridis')
norm = mpl.colors.Normalize(vmin=angles.min(), vmax=angles.max())

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlabel('Energy (eV)')
ax.set_ylabel('Flux')
for angle in angles:
    tally_flux = sp.get_tally(name=f'diamond flux tally {angle:.0f}deg').get_pandas_dataframe()['mean'].to_numpy()
    #variance = sp.get_tally(name=f'diamond flux tally {angle:.0f}deg').get_pandas_dataframe()['std. dev.'].to_numpy() if N_batches > 1 else np.array([0])
    ax.stairs(tally_flux, energies, color=cmap(norm(angle)))
ax.grid()
scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
scalar_map.set_array([])

colorbar = fig.colorbar(scalar_map, ax=ax)
colorbar.set_label("Angle (degrees)")
plt.show()

tot_tally_fluxes = np.zeros([0])
errors = np.zeros([0])

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlabel('Angle (degrees)')
ax.set_ylabel('Total Flux')
for angle in angles:
    tally_flux = sp.get_tally(name=f'diamond total flux tally {angle:.0f}deg').get_pandas_dataframe()['mean'].to_numpy()
    variance = sp.get_tally(name=f'diamond total flux tally {angle:.0f}deg').get_pandas_dataframe()['std. dev.'].to_numpy() if N_batches > 1 else np.array([0])
    ax.errorbar(angle, tally_flux, yerr=variance, fmt='o', color=cmap(norm(angle)))
    errors = np.append(errors, variance)
    tot_tally_fluxes = np.append(tot_tally_fluxes, tally_flux)
ax.grid()
scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
scalar_map.set_array([])

colorbar = fig.colorbar(scalar_map, ax=ax)
colorbar.set_label("Angle (degrees)")
plt.show()