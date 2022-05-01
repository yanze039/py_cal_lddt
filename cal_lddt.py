import argparse
import numpy as np
import os
import MDAnalysis as mda

eps=1e-8

def get_distogram(pdb_file):
    """
    modify select policy to get new distogram
    """
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
    boolean_contact = ref<cutoff
    diff = model[boolean_contact] - ref[boolean_contact]
    contact_num1 = np.sum( np.abs(diff) < 0.5 ) - residue_length
    contact_num2 = np.sum( np.abs(diff) < 1.  ) - residue_length
    contact_num3 = np.sum( np.abs(diff) < 2.  ) - residue_length
    contact_num4 = np.sum( np.abs(diff) < 4.  ) - residue_length
    origin_contact = np.sum( boolean_contact ) - residue_length
    lddt = (contact_num1 + contact_num2 + contact_num3 + contact_num4) / (origin_contact * 4)
    return lddt

def cal_residue_lddt(ref, model, cutoff=15.):
    residue_length = model.shape[0]
    boolean_contact = ref<cutoff
    residue_lddt = np.zeros((residue_length,))
    for resi in range(residue_length):
        diff = model[resi][boolean_contact[resi]] - ref[resi][boolean_contact[resi]]
        contact_num1 = np.sum( np.abs(diff) < 0.5 ) - 1
        contact_num2 = np.sum( np.abs(diff) < 1.  ) - 1
        contact_num3 = np.sum( np.abs(diff) < 2.  ) - 1
        contact_num4 = np.sum( np.abs(diff) < 4.  ) - 1
        origin_contact = np.sum( boolean_contact[resi] ) - 1
        lddt = (contact_num1 + contact_num2 + contact_num3 + contact_num4) / (origin_contact * 4)
        residue_lddt[resi] = lddt
    return residue_lddt

def main(args):
    native_distogram = get_distogram(args.native)
    model_distogram = get_distogram(args.model)
    lddt = cal_lddt(native_distogram, model_distogram, cutoff=args.cutoff)
    per_resi_lddt= cal_residue_lddt(ref, model, cutoff=args.cutoff)
    print("Global lDDT:", lddt)
    print("Per-residue lDDT:\n", lddt)
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

    
