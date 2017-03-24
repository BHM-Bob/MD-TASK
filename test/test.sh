#!/bin/bash


mkdir data
cd data

BIN_DIR=../..

cp $BIN_DIR/example/* .


echo ""
echo "#### BETWEENNESS CENTRALITY - WT ####"
echo ""
echo ""

PREFIX=wt

python $BIN_DIR/calc_network.py --topology $PREFIX.pdb --threshold 7.0 --step 100 --generate-plots --calc-BC --discard-graphs --lazy-load $PREFIX.dcd

mv ${PREFIX}_0_bc.dat ref_${PREFIX}_bc.dat 

python $BIN_DIR/calc_delta_BC.py --generate-plots --reference ref_${PREFIX}_bc.dat --alternatives ${PREFIX}_*_bc.dat
python $BIN_DIR/avg_network.py --data ${PREFIX}_*_bc_delta_BC.dat --data-type delta-BC --prefix ${PREFIX} --generate-plots --x-label "Residues" --y-label "Avg delta BC" --title "Wild Type"


echo ""
echo ""
echo "#### BETWEENNESS CENTRALITY - MUTANT ####"
echo ""
echo ""

PREFIX=mutant

cp $BIN_DIR/examples/* .

python $BIN_DIR/calc_network.py --topology $PREFIX.pdb --threshold 7.0 --step 100 --generate-plots --calc-BC --discard-graphs --lazy-load $PREFIX.dcd

mv ${PREFIX}_0_bc.dat ref_${PREFIX}_bc.dat 

python $BIN_DIR/calc_delta_BC.py --generate-plots --reference ref_${PREFIX}_bc.dat --alternatives ${PREFIX}_*_bc.dat
python $BIN_DIR/avg_network.py --data ${PREFIX}_*_bc_delta_BC.dat --data-type delta-BC --prefix ${PREFIX} --generate-plots --x-label "Residues" --y-label "Avg delta BC" --title "Mutant"


echo ""
echo ""
echo "#### BETWEENNESS CENTRALITY - WT vs MUTANT ####"
echo ""
echo ""
python $BIN_DIR/compare_networks.py --prefix "wt_mutant_avg" --reference-label Wild-type --alternative-label Mutant --y-label "Delta BC" --reference wt_delta_BC_avg.dat --alternative mutant_delta_BC_avg.dat
python $BIN_DIR/compare_networks.py --prefix "wt_mutant_std_dev" --reference-label Wild-type --alternative-label Mutant --y-label "Delta BC" --reference wt_delta_BC_std_dev.dat --alternative mutant_delta_BC_std_dev.dat
python $BIN_DIR/delta_networks.py --reference wt_delta_BC_avg.dat --reference-std wt_delta_BC_std_dev.dat --alternatives *delta_BC_avg.dat --alternatives-std *delta_BC_std_dev.dat --absolute --prefix wt_mutant_cmap --title "My Protein" --x-label "Residues" --y-label "Proteins"


echo ""
echo ""
echo "#### R IGRAPH - CONTACT MAP ####"
echo ""
echo ""
python $BIN_DIR/contact_map.py --residue ASP31 --prefix wt --topology wt.pdb wt.dcd
python $BIN_DIR/contact_map.py --residue ASN31 --prefix mutant --topology mutant.pdb mutant.dcd

echo ""
echo ""
echo "#### TEST PERTURBATION RESPONSE SCANNING ####"
echo ""
echo ""
python $BIN_DIR/prs.py --initial initial.xyz --final final.xyz --perturbations 100 --step 100 --prefix result --topology example_small.pdb example_small.dcd


echo ""
echo ""
echo "#### TEST DYNAMIC CROSS CORRELATION ####"
echo ""
echo ""
python $BIN_DIR/calc_correlation.py --step 100 --prefix example_corr --trajectory example_small.dcd --topology example_small.pdb --lazy-load
