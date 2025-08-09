#!/usr/bin/env python
#
# Perform PRS calculations given and MD trajectory and a final state
# co-ordinate file
#
# Script distributed under GNU GPL 3.0
#
# Author: David Penkler
# Date: 17-11-2016

from math import ceil, floor, log10
from typing import List, Union

import numpy as np
from lazydock_md_task import sdrms
from mbapy_lite.base import put_err, put_log
from mbapy_lite.web import TaskPool
from MDAnalysis import Universe
from tqdm import tqdm


def round_sig(x, sig=2):
    return round(x,sig-int(floor(log10(x)))-1)


def align_frame(reference_frame, alternative_frame):
    n_residues = reference_frame.shape[0]
    return sdrms.superpose3D(alternative_frame.reshape(n_residues, 3), reference_frame)[0].reshape(1, n_residues*3)[0]


def calc_rmsd(reference_frame, alternative_frame):
    return sdrms.superpose3D(alternative_frame, reference_frame)[1]


def run_single_perturbation(n_residues: int, corr_mat: np.ndarray, initial_pose: np.ndarray):
    diffP = np.zeros((n_residues, n_residues*3))
    for i in range(n_residues):
        delF = np.zeros((n_residues*3))
        f = 2 * np.random.random((3, 1)) - 1
        j = (i + 1) * 3

        delF[j-3] = round_sig(abs(f[0,0]), 5)* -1 if f[0,0]< 0 else round_sig(abs(f[0,0]), 5)
        delF[j-2] = round_sig(abs(f[1,0]), 5)* -1 if f[1,0]< 0 else round_sig(abs(f[1,0]), 5)
        delF[j-1] = round_sig(abs(f[2,0]), 5)* -1 if f[2,0]< 0 else round_sig(abs(f[2,0]), 5)

        diffP[i, :] = np.dot((delF), (corr_mat))
        diffP[i, :] = diffP[i, :] + initial_pose.reshape(-1)

        diffP[i, :] = ((sdrms.superpose3D(diffP[i, :].reshape(n_residues, 3), initial_pose)[0].reshape(1, n_residues*3))[0]) - initial_pose.reshape(-1)
    
    return diffP


def main(top_path: str, traj_path: str, chains: List[str], start: int, step: int, stop: int,
         perturbations=250, initial: Union[int, np.ndarray] = 0, final: Union[int, np.ndarray] = -1,
         n_worker = 4):
    # select atoms
    u = Universe(top_path, traj_path)
    mask = u.atoms.names == 'CA'
    chain_mask = u.atoms.chainIDs == chains[0]
    for chain_i in chains[1:]:
        chain_mask = chain_mask | (u.atoms.chainIDs == chain_i)
    ag = u.atoms[mask & chain_mask]
    if len(ag) == 0:
        return put_err(f'No CA atoms found in chains {chains}')
    # load trajectory
    stop = stop or len(u.trajectory)
    sum_frames = ceil((stop - start) / step)
    trajectory = np.zeros((sum_frames, len(ag), 3), dtype=np.float64)
    for current, _ in enumerate(tqdm(u.trajectory[start:stop:step],
                                     desc='Gathering coordinates', total=sum_frames, leave=False)):
        trajectory[current] = ag.positions.astype(np.float64)
    # get initial and final pose
    initial_pose = initial if isinstance(initial, np.ndarray) else trajectory[initial].copy()
    final_pose = final if isinstance(final, np.ndarray) else trajectory[final].copy()
    trajectory.reshape(sum_frames, 3*ag.n_residues)
    put_log('- Final trajectory matrix size: %s\n' % str(trajectory.shape))
    
    put_log("Aligning trajectory frames...\n")
    aligned_mat = np.zeros((sum_frames,3*ag.n_residues))
    frame_0 = trajectory[0].reshape(ag.n_residues, 3)
    for frame in range(0, sum_frames):
        aligned_mat[frame] = align_frame(frame_0, trajectory[frame])

    put_log("- Calculating average structure...\n")
    average_structure_1 = np.mean(aligned_mat, axis=0).reshape(ag.n_residues, 3)

    put_log("- Aligning to average structure...\n")
    for _ in range(0, 10):
        for frame in range(0, sum_frames):
            aligned_mat[frame] = align_frame(average_structure_1, aligned_mat[frame])
        average_structure_2 = np.average(aligned_mat, axis=0).reshape(ag.n_residues, 3)
        rmsd = calc_rmsd(average_structure_1, average_structure_2)
        put_log('   - %s Angstroms from previous structure\n' % str(rmsd))
        average_structure_1 = average_structure_2
        del average_structure_2
        if rmsd <= 0.000001:
            for frame in range(0, sum_frames):
                aligned_mat[frame] = align_frame(average_structure_1, aligned_mat[frame])
            break

    put_log("Calculating difference between frame atoms and average atoms...\n")
    meanstructure = average_structure_1.reshape(ag.n_residues*3)

    put_log('- Calculating R_mat\n')
    R_mat = aligned_mat - meanstructure.reshape(1, -1)
    corr_mat = (R_mat.T @ R_mat) / (sum_frames-1)

    put_log('Calculating experimental difference between initial and final co-ordinates...\n')
    final_alg = sdrms.superpose3D(final_pose, initial_pose)[0]
    diffE = (final_alg-initial_pose).reshape(ag.n_residues, 3)

    put_log(f'Implementing perturbations in parallel with {n_worker} workers...\n')
    diffP = np.zeros((ag.n_residues, ag.n_residues*3, perturbations))
    pool = TaskPool('process', n_worker=n_worker).start()
    for s in tqdm(range(0, perturbations), total=perturbations, desc='perform perturbations', leave=False):
        pool.add_task(s, run_single_perturbation, ag.n_residues, corr_mat.copy(), initial_pose.copy())
        pool.wait_till(lambda: pool.count_waiting_tasks() == 0, 0.01, update_result_queue=False)
    for s in range(0, perturbations):
        diffP[:, :, s] = pool.query_task(s, True, 999)
    pool.close(1)

    # calculate pearson's coefficient
    ## 计算DTarget的向量化版本
    DTarget = np.linalg.norm(diffE, axis=1)
    ## 计算DIFF的向量化版本
    diffP_reshaped = diffP.reshape(ag.n_residues, ag.n_residues, 3, perturbations)
    DIFF = np.linalg.norm(diffP_reshaped, axis=2).transpose(1, 0, 2)
    # 计算RHO的向量化版本
    ## 重组DIFF为二维矩阵便于批量计算
    reshaped_diff = DIFF.transpose(1, 2, 0).reshape(-1, ag.n_residues)
    dt_centered = DTarget - DTarget.mean()
    ## 批量计算协方差和标准差
    diff_centered = reshaped_diff - reshaped_diff.mean(axis=1, keepdims=True)
    covariances = (diff_centered @ dt_centered) / (ag.n_residues - 1)
    std_devs = diff_centered.std(axis=1, ddof=1) * dt_centered.std(ddof=1)
    ## 避免除以零（假设数据无零标准差情况）
    max_RHO: np.ndarray = (covariances / std_devs).reshape(ag.n_residues, perturbations).max(axis=-1)
    return max_RHO
