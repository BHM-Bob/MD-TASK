'''
Date: 2025-02-01 10:35:36
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2025-02-01 11:01:08
Description: 
'''
#! /usr/bin/env python
#
# Calculate SMART RMSD with or without molecular superposition (FIT or NOFIT)
# Script distributed under GNU LGPL 3.0 along rDock software.
#
# Author: Daniel Alvarez-Garcia
# Date: 08-11-2013

import math
#import pybel
import numpy as npy

def superpose3D(ref, target, weights=None,refmask=None,targetmask=None,returnRotMat=False):
    """superpose3D performs 3d superposition using a weighted Kabsch algorithm : http://dx.doi.org/10.1107%2FS0567739476001873 & doi: 10.1529/biophysj.105.066654
    definition : superpose3D(ref, target, weights,refmask,targetmask)
    @parameter 1 :  ref - xyz coordinates of the reference structure (the ligand for instance)
    @type 1 :       float64 numpy array (nx3)
    ---
    @parameter 2 :  target - theoretical target positions to which we should move (does not need to be physically relevant.
    @type 2 :       float 64 numpy array (nx3)
    ---
    @parameter 3:   weights - numpy array of atom weights (usuallly between 0 and 1)
    @type 3 :       float 64 numpy array (n)
    @parameter 4:   mask - a numpy boolean mask for designating atoms to include
    Note ref and target positions must have the same dimensions -> n*3 numpy arrays where n is the number of points (or atoms)
    Returns a set of new coordinates, aligned to the target state as well as the rmsd
    """
    if weights == None :
        weights=1.0
    if refmask == None :
        refmask=npy.ones(len(ref),"bool")
    if targetmask == None :
        targetmask=npy.ones(len(target),"bool")
    #first get the centroid of both states
    ref_centroid = npy.mean(ref[refmask]*weights,axis=0)
    #print ref_centroid
    refCenteredCoords=ref-ref_centroid
    #print refCenteredCoords
    target_centroid=npy.mean(target[targetmask]*weights,axis=0)
    targetCenteredCoords=target-target_centroid
    #print targetCenteredCoords
    #the following steps come from : http://www.pymolwiki.org/index.php/OptAlign#The_Code and http://en.wikipedia.org/wiki/Kabsch_algorithm
    # Initial residual, see Kabsch.
    E0 = npy.sum( npy.sum(refCenteredCoords[refmask] * refCenteredCoords[refmask]*weights,axis=0),axis=0) + npy.sum( npy.sum(targetCenteredCoords[targetmask] * targetCenteredCoords[targetmask]*weights,axis=0),axis=0)
    reftmp=npy.copy(refCenteredCoords[refmask])
    targettmp=npy.copy(targetCenteredCoords[targetmask])
    #print refCenteredCoords[refmask]
    #single value decomposition of the dotProduct of both position vectors
    try:
        dotProd = npy.dot( npy.transpose(reftmp), targettmp* weights)
        V, S, Wt = npy.linalg.svd(dotProd )
    except Exception:
        try:
            dotProd = npy.dot( npy.transpose(reftmp), targettmp)
            V, S, Wt = npy.linalg.svd(dotProd )
        except Exception:
            print("Couldn't perform the Single Value Decomposition, skipping alignment")
        return ref, 0
    # we already have our solution, in the results from SVD.
    # we just need to check for reflections and then produce
    # the rotation.  V and Wt are orthonormal, so their det's
    # are +/-1.
    reflect = float(str(float(npy.linalg.det(V) * npy.linalg.det(Wt))))
    if reflect == -1.0:
        S[-1] = -S[-1]
        V[:,-1] = -V[:,-1]
    rmsd = E0 - (2.0 * sum(S))
    rmsd = npy.sqrt(abs(rmsd / len(ref[refmask])))   #get the rmsd
    #U is simply V*Wt
    U = npy.dot(V, Wt)  #get the rotation matrix
    # rotate and translate the molecule
    new_coords = npy.dot((refCenteredCoords), U)+ target_centroid  #translate & rotate
    #new_coords=(refCenteredCoords + target_centroid)
    #print U
    if returnRotMat :
        return new_coords,rmsd, U
    return new_coords,rmsd


