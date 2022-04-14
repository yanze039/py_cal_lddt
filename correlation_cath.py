import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import MDAnalysis as mda

# from fold.heads import compute_plddt
eps=1e-8

def get_distogram(pdb_file):
    u = mda.Universe(pdb_file)
    ca = u.select_atoms("name CA")
    pos = ca.positions
    dist_matrics = pos.reshape([1,-1,3])-pos.reshape([-1,1,3])
    distogram = np.power(np.sum(dist_matrics ** 2, axis=2), 0.5)
    return distogram

def cal_lddt(ref, model, cutoff=15.):
    """
    ref: Native Structure
    model: Evaluated Structure
    cutoff: unit - angstrom
    """
    residue_length = model.shape[0]
    ref = ref[:residue_length,:residue_length]
    boolean_contact = ref<cutoff
    diff = model[boolean_contact] - ref[boolean_contact]
    contact_num1 = np.sum( np.abs(diff) < 0.5 ) - residue_length
    contact_num2 = np.sum( np.abs(diff) < 1.  ) - residue_length
    contact_num3 = np.sum( np.abs(diff) < 2.  ) - residue_length
    contact_num4 = np.sum( np.abs(diff) < 4.  ) - residue_length
    origin_contact = np.sum( boolean_contact ) - residue_length
    lddt = (contact_num1 + contact_num2 + contact_num3 + contact_num4) / (origin_contact * 4)
    return lddt

def get_native_distogram(case_name):
    disto_path = "/data/yanze/Protein_Score/cath-decoys-20220302-mod/distogram-tst"
    data = np.load(os.path.join(disto_path, case_name+".npy"))
    return data

def main(args):
    native_distogram = get_distogram(args.native)
    model_distogram = get_distogram(args.model)
    lddt = cal_lddt(native_distogram, model_distogram, cutoff=15.)
    print(lddt)
    return lddt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the trained model')
    parser.add_argument('--native', '-n', type=str, default=None,
                        help='Path to Native Structure')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to Model to be evaluated')
    parser.add_argument('--cutoff', '-c', type=float, default=15.0,
                        help='Cutoff, Units: angstrom')
    args = parser.parse_args()
    main(args)

    
