from JP_compiled_source import nGen_diamond
import openmc
#from fng_source import FNGSource
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import  skewnorm
import pickle

def create_parallel_planes(plane, distance):
    """
    Create two parallel planes at equal distance from the input plane.
    
    Parameters:
    -----------
    plane : openmc.Plane
        The input plane with equation: a*x + b*y + c*z = d
    distance : float
        The perpendicular distance from the input plane to each output plane
    
    Returns:
    --------
    tuple of (plane1, plane2) : openmc.Plane
        Two planes parallel to input, at +distance and -distance respectively
    
    Example:
    --------
    original = create_plane_from_y_axis(45)
    plane_inner, plane_outer = create_parallel_planes(original, 0.1)
    """
    
    # Extract plane coefficients
    a, b, c, d = plane.a, plane.b, plane.c, plane.d
    
    # Calculate the magnitude of the normal vector
    normal_magnitude = np.sqrt(a**2 + b**2 + c**2)
    
    # Calculate the offset for d to achieve the desired perpendicular distance
    d_offset = distance * normal_magnitude
    
    # Create two parallel planes
    plane1 = openmc.Plane(a=a, b=b, c=c, d=d + d_offset, name=f'{plane.name}_offset_pos')
    plane2 = openmc.Plane(a=a, b=b, c=c, d=d - d_offset, name=f'{plane.name}_offset_neg')
    
    return plane1, plane2

def create_plane_from_y_axis(angle_degrees):
    """
    Create a plane containing the z-axis at a given angle from the y-axis.
    
    Parameters:
    -----------
    angle_degrees : float
        Angle in degrees from the y-axis (measured counterclockwise in xy-plane)
    
    Returns:
    --------
    openmc.Plane
        A plane containing the z-axis at the specified angle
    
    Example:
    --------
    plane_at_45 = create_plane_from_y_axis(45)  # Plane at 45° from y-axis
    """
    angle_degrees_mod = np.abs(angle_degrees) # Normalize angle to [0, 360)
    # Convert angle to radians
    if angle_degrees >= 0:
        a = -np.cos(np.radians(angle_degrees_mod))
        b = np.sin(np.radians(angle_degrees_mod))
    else:
        a = np.cos(np.radians(angle_degrees_mod))
        b = np.sin(np.radians(angle_degrees_mod))
    
    # Calculate normal vector components (in xy-plane, perpendicular to z-axis)
    # Normal vector: (sin(θ), cos(θ), 0)
    # a = np.cos(angle_rad)
    # b = np.sin(angle_rad)
    c = 0
    
    # Create plane with equation: a*x + b*y + c*z = 0 (passes through origin)
    plane = openmc.Plane(a=a, b=b, c=c, d=0, name=f'Plane_{angle_degrees}deg')
    
    return plane

Diamond = openmc.Material(name='Diamond')
Diamond.add_element('C', 1, 'ao')
Diamond.set_density('g/cm3', 3.51)

Air = openmc.Material(name='Air')
Air.add_element('N', 0.78, 'ao')
Air.add_element('O', 0.22, 'ao')
Air.set_density('g/cm3', 0.001225)

Materials = openmc.Materials([Diamond, Air])

World = -openmc.Sphere(r=20.0, name='World', boundary_type='vacuum')

Diamond_length = 0.8
Diamond_width = 0.4
Diamond_thickness = 0.05
Distance_from_source = 8.0 # cm, distance from source to front face of diamond detector

outer_cylinder = openmc.ZCylinder(x0=0, y0=0, r=Distance_from_source+Diamond_thickness, name='Outer cylinder')
inner_cylinder = openmc.ZCylinder(x0=0, y0=0, r=Distance_from_source, name='Inner cylinder')

upper_plane = openmc.ZPlane(z0=Diamond_length/2, name='Upper plane')
lower_plane = openmc.ZPlane(z0=-Diamond_length/2, name='Lower plane')

diamond_regions = {}
diamond_cells = {}
diamond_cell_array = []

All_diamond_regions = None


for i,angle in enumerate(np.arange(-135, 135.1, 9)):
    plane = create_plane_from_y_axis(angle)
    if angle >= 0:
        half_plane = create_plane_from_y_axis(angle + 10)
    else:  # Perpendicular plane to define half-space
        half_plane = create_plane_from_y_axis(angle - 10)
    plane_inner, plane_outer = create_parallel_planes(plane, Diamond_width/2)
    print(f"Angle: {angle}° -> Plane: {plane.name}, Inner: {plane_inner.name}, Outer: {plane_outer.name}")

    diamond_region = +plane_outer & -plane_inner & +half_plane & -outer_cylinder & +inner_cylinder & -upper_plane & +lower_plane
    #+plane_inner & -plane_outer 
    #& -outer_cylinder & +inner_cylinder & -upper_plane & +lower_plane
    diamond_regions[angle] = diamond_region

    diamond_cells[angle] = openmc.Cell(name=f'Diamond cell {angle}deg', region=diamond_region, fill=Diamond)
    All_diamond_regions = diamond_regions[angle] if All_diamond_regions is None else All_diamond_regions | diamond_regions[angle]

    diamond_cell_array.append(diamond_cells[angle])

World = World & ~All_diamond_regions 

air_cell = openmc.Cell(name='Air cell', region=World, fill=Air)

MyGeometry = openmc.Geometry(diamond_cell_array + [air_cell])

MyGeometry.export_to_xml()
Materials.export_to_xml()


Energy_edges = np.arange(13.0e6, 15.6e6, 2e4) # eV
Energy_filter = openmc.EnergyFilter(Energy_edges) # eV
Energy_filter2 = openmc.EnergyFilter([13.0e6, 15.6e6]) # eV just taking the entire dt fusion range

diamond_flux_tallies = []

for i in range(len(diamond_cell_array)):
    tally = openmc.Tally(name=f'diamond flux tally {-135 + i*9}deg')
    tally_2 = openmc.Tally(name=f'diamond total flux tally {-135 + i*9}deg')
    diamond_filter = openmc.CellFilter([diamond_cell_array[i]])
    tally.filters = [diamond_filter, Energy_filter]  
    tally.scores = ['flux']
    tally_2.filters = [diamond_filter, Energy_filter2]
    tally_2.scores = ['flux']
    diamond_flux_tallies.append(tally)
    diamond_flux_tallies.append(tally_2)

mytallies = openmc.Tallies(diamond_flux_tallies)
#mytallies.export_to_xml()

# Create a geometry plot of the xy plane
plot = openmc.Plot()
plot.basis = 'xy'  # View the xy plane
plot.origin = (0, 0, 0)
plot.width = (30, 30)  # Width in x and y directions (cm)
plot.color_by = 'material'  # Color by material
plot.pixels = (1000, 1000)  # Resolution

# # Create a PlotCollection and export
plots = openmc.Plots([plot])
plots.export_to_xml()

# Generate the plot image
openmc.plot_geometry()

# Run the OpenMC simulation
N_particles = 1000000 # If we do 50 million, the flux tally will be over 5000 on each diamond in theory
N_batches = 5

source = nGen_diamond(N_char=2, center=(0, 0, 0), reference_uvw=(0, 1, 0), Show_plots=False)

run_dir = f'results_{int(N_particles)}_particles_{int(N_batches)}_batches'

settings = openmc.Settings()
settings.source = source
settings.particles = N_particles
settings.batches = N_batches
settings.run_mode = "fixed source"

model = openmc.Model(geometry=MyGeometry, materials=Materials, settings=settings, tallies=mytallies)
model.run(cwd=run_dir, threads=16, geometry_debug=False)
