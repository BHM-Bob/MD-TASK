#!/usr/bin/env python

"""
Generates weighted contact map around a given residue from an MD trajectory
"""

import sys
import argparse
from datetime import datetime
import numpy as np
import mdtraj as md
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from lazydock_md_task.utils import format_seconds

__author__ = "Olivier Sheik Amamuddy"
__copyright__ = "Copyright 2019, Research Unit in Bioinformatics"
__credits__ = ["MDTraj developers"]
__license__ = "GNU GPL 3.0"
__version__ = "1.1.0"
__maintainer__ = "Olivier Sheik Amamuddy"
__email__ = "oliserand@gmail.com"
__status__ = "Production"
__date__ = "23rd October 2019"

def plot_network(g, ebunch, contact_map, discardplot=False, node_size=2900,
                 node_fontsize=9.5, edgewidth_factor=10, edgelabel_fontsize=8):
    """
    Plots a network
    """
    tmp = pd.DataFrame(ebunch)
    order = pd.Series([int(i[3:-2]) for i in tmp.loc[:, 1].tolist()]).sort_values().index
    ebunch = tmp.iloc[order, :].values
    edges = g.edges()
    node_colors = []
    for node in g.nodes():
        node_colors.append('#B1DF61')
    if discardplot is False:
        plt.figure(figsize=[6.5, 6.5])
        layout = nx.layout.shell_layout(g, nlist=[ebunch[0], [i[1] for i in ebunch]])
        edgewidths = [g[i][j]['weight']*edgewidth_factor for i, j in edges]

        nx.draw_networkx(g, pos=layout, edgelist=edges, node_color=node_colors,
                         edge_color='lightgrey', node_size=node_size,
                         width=edgewidths, with_labels=True, font_family='serif',
                         font_size=node_fontsize)
        nx.draw_networkx_edge_labels(g, pos=layout, alpha=1,
                                     font_size=edgelabel_fontsize,
                                     font_family='serif',
                                     bbox={'facecolor':'white', 'alpha':0},
                                     edge_labels={(i, j): round(g[i][j]['weight'], 3)
                                                  for i, j, k in ebunch},
                                     rotate=False)
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(contact_map, format='pdf')

def main(args):
    """
    Main function
    """
    traj_path = args.trajectory
    topology = args.topology
    cutoff = args.threshold / 10
    chain = args.chain
    node_size = args.nodesize
    node_fontsize = args.nodefontsize
    edgewidth_factor = args.edgewidthfactor
    edgelabel_fontsize = args.edgelabelfontsize
    discardplot = args.discard_graphs

    if args.residue is not None:
        residue = args.residue.upper()
    else:
        log("A residue has to be provided. Try -h option.\n")
        sys.exit()

    prefix = residue
    chain_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    log("Loading trajectory...\n")

    try:
        traj = md.load(traj_path, top=topology)[::args.step]
    except TypeError as ex:
        print(ex)
        sys.exit()

    atom_indices = [atom.index for atom in traj.topology.atoms if
                    ((atom.name == "CB" and atom.residue.name != "GLY") or \
        (atom.name == "CA" and atom.residue.name == "GLY"))]
    traj = traj.atom_slice(atom_indices, inplace=True)
    atom_indices = [atom.index for atom in traj.topology.atoms]
    residues = list(map(lambda x: str(x), traj.top.residues))
    if residue not in residues:
        log("ERROR: Residue {} not found.\n".format(residue))
        sys.exit()
    nframes = traj.n_frames
    center = "{}.{}".format(residue, chain)
    contacts = {}

    log("Calculating weighted contacts around %s (chain %s)...\n" % (residue, chain))

    for frame in traj:
        for aaidx, atom1_idx in enumerate(atom_indices):
            if residues[aaidx] == residue:
                atom1 = list(frame.top.atoms)[atom1_idx]
                atom1_chain = chain_chars[atom1.residue.chain.index]
                atom1_resname = atom1.residue.name
                atom1_resid = atom1.residue.resSeq
                if atom1_chain == chain:
                    for atom2_idx in atom_indices:
                        if atom1_idx != atom2_idx:
                            distance = np.linalg.norm(frame.xyz[0, atom1_idx] \
                                                      - frame.xyz[0, atom2_idx])
                            if distance < cutoff:
                                atom2 = list(frame.top.atoms)[atom2_idx]
                                atom2_chain = chain_chars[atom2.residue.chain.index]
                                atom2_resname = atom2.residue.name
                                atom2_resid = atom2.residue.resSeq
                                edge = ("{}{}.{}".format(atom1_resname, atom1_resid, atom1_chain),
                                        "{}{}.{}".format(atom2_resname, atom2_resid, atom2_chain))
                                if edge not in contacts.keys():
                                    contacts[edge] = 1
                                else:
                                    contacts[edge] += 1
                    break

    if args.ocsv is not None:
        csv_file = args.ocsv
    else:
        csv_file = "%s_chain%s_network.csv" % (prefix, chain)

    if args.opdf is not None:
        contact_map = args.opdf
    else:
        contact_map = "%s_chain%s_contact_map.pdf" % (prefix, chain)

    log("Generating contact map: %s...\n" % contact_map)

    _ = nx.Graph()
    ebunch = [[center, x[0][1], round(x[1]/float(nframes), 3)] for x in list(contacts.items())]
    _.add_weighted_edges_from(ebunch)

    plot_network(_, ebunch, contact_map, discardplot=discardplot,
                 node_size=node_size, node_fontsize=node_fontsize,
                 edgewidth_factor=edgewidth_factor,
                 edgelabel_fontsize=edgelabel_fontsize)

    log("Writing network to %s...\n" % csv_file)

    dframe = pd.DataFrame(ebunch)
    dframe.to_csv(csv_file, header=False, index=False)

SILENT = False
STREAM = sys.stdout

def log(message):
    """
    Displays messages to stdout
    """
    global SILENT, STREAM
    if not SILENT:
        STREAM.write(message)

def parse_args():
    """
    Argument parser
    """
    parser = argparse.ArgumentParser(description="Generates weighted contact \
                                     map around a given residue from an MD \
                                     trajectory")
    parser.add_argument("--silent", help="Turn off logging", action='store_true',
                        default=False)
    parser.add_argument("--log-file", help="Output log file (default: standard output)",
                        default=None)
    parser.add_argument("trajectory", help="Trajectory file")
    parser.add_argument("--topology",
                        help="Topology PDB file (required if trajectory does \
                        not contain topology information)")
    parser.add_argument("--residue",
                        help="The residue that the contact map will be built \
                        around (e.g. THR405)")
    parser.add_argument("--threshold",
                        help="Maximum distance threshold in Angstroms when \
                        constructing graph (default: 6.7 Angstroms)",
                        default=6.7, type=float)
    parser.add_argument("--ocsv", help="Name of the CSV contact file",
                        default=None)
    parser.add_argument("--opdf", help="Name of the PDF contact file",
                        default=None)
    parser.add_argument("--step",
                        help="Size of step when iterating through trajectory frames",
                        default=1, type=int)
    parser.add_argument("--chain", help="Chain ID to be matched (default: A)",
                        default="A")
    parser.add_argument("--discard-graphs",
                        help="Suppress plotting. Only produce the CSV contact file",
                        action='store_true')
    parser.add_argument("--nodesize", help="The node size (default:2900)",
                        default=2900, type=int)
    parser.add_argument("--nodefontsize", help="The node font size (default:9.5)",
                        default=9.5, type=float)
    parser.add_argument("--edgewidthfactor",
                        help="Scaling factor for the plotted edge thickness (default:10.0)",
                        default=10.0, type=float)
    parser.add_argument("--edgelabelfontsize",
                        help="Font size for the edge labels (default:8.0)",
                        default=8.0, type=float)
    args = parser.parse_args()
    return  args

if __name__ == "__main__":
    ARGS = parse_args()
    SILENT = ARGS.silent

    if ARGS.log_file is not None:
        STREAM = open(ARGS.log_file, 'w')

    START = datetime.now()
    log("Started at: %s\n" % str(START))
    main(ARGS)
    END = datetime.now()
    TIME_TAKEN = format_seconds((END - START).seconds)
    log("Completed at: %s\n" % str(END))
    log("- Total time: %s\n" % str(TIME_TAKEN))
    STREAM.close()
