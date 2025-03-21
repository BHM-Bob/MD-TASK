#!/usr/bin/env python
#
# Perform PRS calculations given and MD trajectory and a final state
# co-ordinate file
#
# Script distributed under GNU GPL 3.0
#
# Author: David Penkler
# Date: 17-11-2016

import argparse
import sys
from math import floor, log10, sqrt

import mdtraj as md
import numpy
from lazydock_md_task import sdrms
from lazydock_md_task.cli import CLI
from lazydock_md_task.trajectory import load_trajectory
from mbapy_lite.base import put_log
from tqdm import tqdm


def round_sig(x, sig=2):
    return round(x,sig-int(floor(log10(x)))-1)


def trajectory_to_array(traj, totalframes, totalres):
    trajectory = numpy.zeros((totalframes, totalres*3))

    for row, frame in enumerate(traj):
        top = frame.topology

        col = 0
        for atom_index, atom in enumerate(top.atoms):
            if atom.name == "CA":
                trajectory[row,col:col+3] = frame.xyz[0,atom_index]*10
                col += 3

    return trajectory


def align_frame(reference_frame, alternative_frame, aln=False):
    totalres = reference_frame.shape[0]

    if aln:
        return sdrms.superpose3D(alternative_frame.reshape(totalres, 3), reference_frame, refmask=mask, targetmask=mask)[0].reshape(1, totalres*3)[0]
    else:
        return sdrms.superpose3D(alternative_frame.reshape(totalres, 3), reference_frame)[0].reshape(1, totalres*3)[0]


def calc_rmsd(reference_frame, alternative_frame, aln=False):
    if aln:
        return sdrms.superpose3D(alternative_frame, reference_frame, refmask=mask, targetmask=mask)[1]
    else:
        return sdrms.superpose3D(alternative_frame, reference_frame)[1]


def main(args):
    from MDAnalysis import Universe
    u = Universe(args.topology, args.trajectory)
    idx = u.select_atoms('protein and name CA').ids
    
    final = md.load_frame(args.trajectory, args.final, top=args.topology, atom_indices=idx)
    final_pos = final[0].xyz[0] * 10.

    initial = md.load_frame(args.trajectory, args.initial, top=args.topology, atom_indices=idx)
    initial_pos = initial[0].xyz[0] * 10.


    put_log("Loading trajectory...\n")

    if args.num_frames:
        traj, totalframes = load_trajectory(args.trajectory, args.topology, args.step, True)
        totalframes = args.num_frames
    else:
        traj, totalframes = load_trajectory(args.trajectory, args.topology, args.step, False)

    totalres = initial.n_residues

    put_log('- Total number of frames = %d\n- Number of residues = %d\n' % (totalframes, totalres))

    trajectory = trajectory_to_array(traj, totalframes, totalres)

    put_log('- Final trajectory matrix size: %s\n' % str(trajectory.shape))
    del traj


    put_log("Aligning trajectory frames...\n")

    aligned_mat = numpy.zeros((totalframes,3*totalres))
    frame_0 = trajectory[0].reshape(totalres, 3)

    for frame in range(0, totalframes):
        aligned_mat[frame] = align_frame(frame_0, trajectory[frame], args.aln)

    del trajectory


    put_log("- Calculating average structure...\n")

    average_structure_1 = numpy.mean(aligned_mat, axis=0).reshape(totalres, 3)


    put_log("- Aligning to average structure...\n")

    for i in range(0, 10):
        for frame in range(0, totalframes):
            aligned_mat[frame] = align_frame(average_structure_1, aligned_mat[frame], args.aln)

        average_structure_2 = numpy.average(aligned_mat, axis=0).reshape(totalres, 3)

        rmsd = calc_rmsd(average_structure_1, average_structure_2, args.aln)

        put_log('   - %s Angstroms from previous structure\n' % str(rmsd))

        average_structure_1 = average_structure_2
        del average_structure_2

        if rmsd <= 0.000001:
            for frame in range(0, totalframes):
                aligned_mat[frame] = align_frame(average_structure_1, aligned_mat[frame], args.aln)
            break


    put_log("Calculating difference between frame atoms and average atoms...\n")

    meanstructure = average_structure_1.reshape(totalres*3)

    del average_structure_1

    put_log('- Calculating R_mat\n')
    R_mat = numpy.zeros((totalframes, totalres*3))
    for frame in range(0, totalframes):
        R_mat[frame,:] = (aligned_mat[frame,:]) - meanstructure

    put_log('- Transposing\n')

    RT_mat = numpy.transpose(R_mat)

    RT_mat = numpy.mat(RT_mat)
    R_mat = numpy.mat(R_mat)

    put_log('- Calculating corr_mat\n')

    corr_mat = (RT_mat * R_mat)/ (totalframes-1)

    put_log('Reading initial and final PDB co-ordinates...\n')
    initial, final = initial_pos, final_pos

    put_log('Calculating experimental difference between initial and final co-ordinates...\n')

    if args.aln:
        put_log("- Using NTD alignment restrictions\n")
        final_alg = sdrms.superpose3D(final, initial, refmask=mask, targetmask=mask)[0]
    else:
        final_alg = sdrms.superpose3D(final, initial)[0]

    diffE = (final_alg-initial).reshape(totalres*3, 1)

    del final
    del final_alg


    put_log('Implementing perturbations sequentially...\n')

    perturbations = int(args.perturbations)
    diffP = numpy.zeros((totalres, totalres*3, perturbations))
    initial_trans = initial.reshape(1, totalres*3)

    for s in tqdm(range(0, perturbations), total=perturbations, desc='perform perturbations', leave=False):
        for i in range(0, totalres):
            delF = numpy.zeros((totalres*3))
            f = 2 * numpy.random.random((3, 1)) - 1
            j = (i + 1) * 3

            delF[j-3] = round_sig(abs(f[0,0]), 5)* -1 if f[0,0]< 0 else round_sig(abs(f[0,0]), 5)
            delF[j-2] = round_sig(abs(f[1,0]), 5)* -1 if f[1,0]< 0 else round_sig(abs(f[1,0]), 5)
            delF[j-1] = round_sig(abs(f[2,0]), 5)* -1 if f[2,0]< 0 else round_sig(abs(f[2,0]), 5)

            diffP[i,:,s] = numpy.dot((delF), (corr_mat))
            diffP[i,:,s] = diffP[i,:,s] + initial_trans[0]

            if args.aln:
                diffP[i,:,s] = ((sdrms.superpose3D(diffP[i,:,s].reshape(totalres, 3), initial, refmask=mask, targetmask=mask)[0].reshape(1, totalres*3))[0]) - initial_trans[0]
            else:
                diffP[i,:,s] = ((sdrms.superpose3D(diffP[i,:,s].reshape(totalres, 3), initial)[0].reshape(1, totalres*3))[0]) - initial_trans[0]
            del delF

    del initial_trans
    del initial
    del corr_mat


    put_log("Calculating Pearson's correlations coefficient...\n")

    DTarget = numpy.zeros(totalres)
    DIFF = numpy.zeros((totalres, totalres, perturbations))
    RHO = numpy.zeros((totalres, perturbations))

    for i in range(0, totalres):
        DTarget[i] = sqrt(diffE[3*(i+1)-3]**2 + diffE[3*(i+1)-2]**2 + diffE[3*(i+1)-1]**2)

    for j in tqdm(range(0, perturbations), total=perturbations, desc='calcu DIFF', leave=False):
        for i in range(0, totalres):
            for k in range(0, totalres):
                DIFF[k,i,j] = sqrt((diffP[i, 3*(k+1)-3, j]**2) + (diffP[i, 3*(k+1)-2, j]**2) + (diffP[i, 3*(k+1)-1, j]**2))

    del diffP

    for i in tqdm(range(0, perturbations), total=perturbations, desc='calcu RHO', leave=False):
        for j in range(0, totalres):
            RHO[j,i] = numpy.corrcoef(numpy.transpose(DIFF[:,j,i]), DTarget)[0,1]

    del DIFF
    del DTarget

    maxRHO = numpy.zeros(totalres)
    for i in range(0, totalres):
        maxRHO[i] = numpy.amax(abs(RHO[i,:]))

    numpy.savetxt("%s.csv" % args.prefix, maxRHO, delimiter=",", header=args.prefix)

    return maxRHO



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("trajectory", help="Trajectory file")
    parser.add_argument("--topology", help="Topology PDB file (required if trajectory does not contain topology information)")
    parser.add_argument("--step", help="Size of step when iterating through trajectory frames", default=1, type=int)
    parser.add_argument("--initial", type=int, help="Initial state co-ordinate file (default: generated from first frame of trajectory)", default=None)
    parser.add_argument("--final", type=int, help="Final state co-ordinate file (must be provided)")
    parser.add_argument("--perturbations", help="Number of perturbations (default: 250)", type=int, default=250)
    parser.add_argument("--num-frames", help="The number of frames in the trajectory (provides improved performance for large trajectories that cannot be loaded into memory)", type=int, default=None)
    parser.add_argument("--aln", help="Restrict N-Terminal alignment", action="store_true")
    parser.add_argument("--prefix", help="Prefix for CSV output file (default: result)", default="result")

    args = parser.parse_args()
    main(args)
