import argparse

import numpy as np

from trizod.constants import AA1TO3, BBATNS
from trizod.potenci.potenci import getpredshifts
from trizod.scoring.scoring import compute_pscores, convert_to_triplet_data, get_offset_corrected_wSCS


def compute_gscores(
    cs: dict[tuple[int, str], float],  # {(res_num, atom_type): chemical_shift_value}
    sequence: str,
    temperature: float = 289.0,
    pH: float = 7.0,
    ionic_strength: float = 0.1,
) -> np.ndarray:
    """Compute g-scores from chemical shifts"""
    #### default parameters ###
    offset_correction = True
    max_offset = 3.0
    reject_shift_type_only = False
    ###########################

    shifts = []
    for (res, atom), val in cs.items():
        shifts.append(("None", "None", str(res), AA1TO3[sequence[res - 1]], atom, atom[0], val, "0.0", ""))
    random_coil_cs = getpredshifts(seq=sequence, temperature=temperature, pH=pH, ion=ionic_strength, pkacsvfile=False)
    ret = get_offset_corrected_wSCS(seq=sequence, shifts=shifts, predshiftdct=random_coil_cs)
    shw, ashwi, cmp_mask, olf, offf, shw0, ashwi0, ol0, off0 = ret
    offsets = offf
    if offset_correction == False:
        ashwi = ashwi0
        offsets = off0
    elif not (max_offset == None or np.isinf(max_offset)):
        # check if any offsets are too large
        for i, at in enumerate(BBATNS):
            if np.abs(offf[at]) > max_offset:
                offsets[at] = np.nan
                if reject_shift_type_only:
                    # mask data related to this backbone shift type, excluding it from scores computation
                    cmp_mask[:, i] = False
    if np.any(cmp_mask):
        ashwi3, k3 = convert_to_triplet_data(ashwi, cmp_mask)
        scores = compute_pscores(ashwi3, k3, cmp_mask)
    else:
        scores = np.full(len(sequence), np.nan)
    return scores


def parse_cs(filename: str) -> dict[tuple[int, str], float]:
    """Parse chemical shifts from a file"""
    atom_types = None
    cs = {}
    with open(filename) as f:
        for line in f:
            line_split = line.split()
            if not atom_types:
                assert len(line_split) > 2, "experimental CS file header too short"
                assert line_split[0] == "#RESID", "expected '#RESID' as first header column"
                assert line_split[1] == "RESNAME", "expected 'RESNAME' as second header column"
                atom_types = line_split[2:]
            else:
                for i, val in enumerate(line_split[2:]):
                    if val.lower() != "nan" and float(val) > 0:
                        cs[(int(line_split[0]), atom_types[i])] = float(val)
    return cs


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-s", "--sequence", type=str, required=True, help="protein sequence")
    argparser.add_argument("-f", "--filename", type=str, required=True, help="file with experimental chemical shifts")
    argparser.add_argument("--temperature", type=float, default=289.0, help="temperature (K)")
    argparser.add_argument("--pH", type=float, default=7.0, help="pH value")
    argparser.add_argument("--ionic_strength", type=float, default=0.1, help="ionic strength (M)")
    args = argparser.parse_args()

    cs = parse_cs(args.filename)
    gscores = compute_gscores(
        cs=cs,
        sequence=args.sequence,
        temperature=args.temperature,
        pH=args.pH,
        ionic_strength=args.ionic_strength,
    )
    print(gscores)
