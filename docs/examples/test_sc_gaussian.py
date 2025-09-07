import argparse
import cheetah
import torch
from scipy import constants


# Settings
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--rms", type=float, default=0.010)
parser.add_argument("--nparts", type=int, default=128_000)
parser.add_argument("--intensity", type=float, default=5e10)
parser.add_argument("--kin-energy", type=float, default=2.5e6)
parser.add_argument("--length", type=float, default=1.0)
parser.add_argument("--nslice", type=int, default=100)
parser.add_argument("--species", type=str, default="proton")
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

torch.set_default_dtype(torch.float64)


# Synchronous particle
# ------------------------------------------------------------------------------

species = cheetah.particles.Species(args.species)
kin_energy = torch.tensor(args.kin_energy)
rest_energy = species.mass_kg * constants.speed_of_light**2 / constants.elementary_charge
energy = kin_energy + rest_energy
gamma = energy / rest_energy
beta = torch.sqrt(1.0 - (1.0 / gamma**2))


# Beam
# ------------------------------------------------------------------------------

particles = torch.zeros((args.nparts, 7))
particles[:, 0] = torch.randn(args.nparts) * args.rms
particles[:, 2] = torch.randn(args.nparts) * args.rms
particles[:, 4] = torch.randn(args.nparts) * args.rms / gamma / beta
particles[:, -1] = torch.ones(args.nparts)

intensity = torch.tensor(args.intensity)
macrosize = intensity / particles.shape[0]
particle_charges = macrosize * constants.elementary_charge

beam = cheetah.ParticleBeam(
    particles,
    energy=energy,
    species=species,
    particle_charges=particle_charges,
)


# Lattice
# ------------------------------------------------------------------------------

nslice = args.nslice
length = torch.tensor(args.length)
slice_length = length / nslice

elements = []
for index in range(nslice):
    elements.append(
        cheetah.SpaceChargeKick(
            slice_length,
            grid_shape=(64, 64, 64),
            grid_extent_x=torch.tensor(3.0),
            grid_extent_y=torch.tensor(3.0),
            grid_extent_tau=torch.tensor(3.0),
        )
    )
    elements.append(cheetah.Drift(slice_length))

segment = cheetah.Segment(elements)


# Track
# ------------------------------------------------------------------------------

particles_in = beam.particles.clone()

for index, element in enumerate(elements):
    beam = element.track(beam)

    with torch.no_grad():
        xrms = 1000.0 * torch.std(beam.particles[:, 0])
        yrms = 1000.0 * torch.std(beam.particles[:, 2])
        zrms = 1000.0 * torch.std(beam.particles[:, 4]) * (gamma * beta)

        message = "step={} xrms={:0.3f} yrms={:0.3f} zrms={:0.3f}".format(
            index,
            xrms,
            yrms,
            zrms,
        )
        print(message)

particles_out = beam.particles.clone()


# Analyze
# ------------------------------------------------------------------------------

if args.plot:
    import matplotlib.pyplot as plt

    limits = 2 * [(-xrms, xrms)]
    limits = torch.tensor(limits)
    limits = 3.0 * limits

    fig, axs = plt.subplots(ncols=2, figsize=(6.0, 2.5), sharex=True, sharey=True)
    for ax, particles in zip(axs, [particles_in, particles_out]):
        ax.hist2d(
            1000.0 * particles[:, 0], 
            1000.0 * particles[:, 4] * gamma * beta, 
            bins=64, 
            range=limits, 
            density=True
        )
    axs[0].set_xlabel("x [mm]")
    axs[1].set_ylabel("z [mm]")
    plt.show()
    
    fig, axs = plt.subplots(ncols=2, figsize=(6.0, 2.5), sharex=True, sharey=True)
    for ax, particles in zip(axs, [particles_in, particles_out]):
        ax.hist2d(
            1000.0 * particles[:, 0], 
            1000.0 * particles[:, 2],
            bins=64, 
            range=limits, 
            density=True
        )
    axs[0].set_xlabel("x [mm]")
    axs[1].set_ylabel("y [mm]")
    plt.show()

