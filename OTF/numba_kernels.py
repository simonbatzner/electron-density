import numpy as np
import scipy as sp
import MD_Parser
import kern_help
from math import exp
from numba import njit
import time


def species_to_bonds():
    pass


class Two_Body_Environment:
    pass


class Two_Body_Env_Old:
    def __init__(self, noa):
        self.coordinates = np.zeros([noa, 3])
        self.relative_coordinates = np.zeros([noa, 3])
        self.distances = np.zeros(noa)
        self.types = np.zeros(noa)
        self.central_atom = 0

    def dict_to_class(self, chem_env, type_conv):

        noa = len(chem_env['dists'])
        self.central_atom = type_conv[chem_env['central_atom']]

        for n in range(noa):
            self.coordinates[n, 0] = chem_env['xs'][n]
            self.coordinates[n, 1] = chem_env['ys'][n]
            self.coordinates[n, 2] = chem_env['zs'][n]

            self.relative_coordinates[n, 0] = chem_env['xrel'][n]
            self.relative_coordinates[n, 1] = chem_env['yrel'][n]
            self.relative_coordinates[n, 2] = chem_env['zrel'][n]

            self.distances[n] = chem_env['dists'][n]

            self.types[n] = type_conv[chem_env['types'][n]]


# get two body kernel between chemical environments
@njit(cache=True)
def two_body_jit(rel1, rel2, dists1, dists2, cent1, cent2, typs1,
                 typs2, d1, d2, sig, ls):
    d = sig*sig/(ls*ls*ls*ls)
    e = ls*ls
    f = 1/(2*ls*ls)
    kern = 0

    x1_len = len(typs1)
    x2_len = len(typs2)

    for m in range(x1_len):
        e1 = typs1[m]
        r1 = dists1[m]
        coord1 = rel1[m, d1]

        for n in range(x2_len):
            e2 = typs2[n]
            r2 = dists2[n]
            coord2 = rel2[n, d2]

            # check that atom types match
            if (cent1 == cent2 and e1 == e2) or (cent1 == e2 and cent2 == e1):
                rr = (r1-r2)*(r1-r2)
                kern += d*exp(-f*rr)*coord1*coord2*(e-rr)

    return kern


# get two body kernel between chemical environments
def two_body(x1, x2, d1, d2, sig, ls):
    d = sig*sig/(ls*ls*ls*ls)
    e = ls*ls
    f = 1/(2*ls*ls)
    kern = 0

    # record central atom types
    c1 = x1['central_atom']
    c2 = x2['central_atom']

    x1_len = len(x1['dists'])
    x2_len = len(x2['dists'])

    for m in range(x1_len):
        e1 = x1['types'][m]
        r1 = x1['dists'][m]
        coord1 = x1[d1][m]
        for n in range(x2_len):
            e2 = x2['types'][n]
            r2 = x2['dists'][n]
            coord2 = x2[d2][n]

            # check that atom types match
            if (c1 == c2 and e1 == e2) or (c1 == e2 and c2 == e1):
                rr = (r1-r2)*(r1-r2)
                kern += d*exp(-f*rr)*coord1*coord2*(e-rr)

    return kern

if __name__ == '__main__':
    outfile = '/Users/jonpvandermause/Research/GP/Datasets/SiC_MD/sic_md.out'
    Si_MD_Parsed = MD_Parser.parse_qe_pwscf_md_output(outfile)

    # set crystal structure
    dim = 3
    alat = 4.344404578
    unit_cell = [[0.0, alat/2, alat/2], [alat/2, 0.0, alat/2],
                 [alat/2, alat/2, 0.0]]  # fcc primitive cell
    unit_pos = [['Si', [0, 0, 0]], ['Si', [alat/4, alat/4, alat/4]]]
    brav_mat = np.array([[0.0, alat/2, alat/2], [alat/2, 0.0, alat/2],
                        [alat/2, alat/2, 0.0]])*dim
    brav_inv = np.linalg.inv(brav_mat)

    # bravais vectors
    vec1 = brav_mat[:, 0].reshape(3, 1)
    vec2 = brav_mat[:, 1].reshape(3, 1)
    vec3 = brav_mat[:, 2].reshape(3, 1)

    # get chemical environments (stored as dictionaries)
    cutoff = 3.6
    pos = Si_MD_Parsed[1]['positions']
    typs = Si_MD_Parsed[1]['elements']
    fcs = kern_help.fc_conv(Si_MD_Parsed[2]['forces'])
    envs = kern_help.get_envs(pos, typs, brav_mat, brav_inv, vec1, vec2,
                              vec3, cutoff)

    # calculate the two body kernel
    x1 = envs[0]
    x2 = envs[1]
    d1 = 'xrel'
    d2 = 'yrel'
    sig = 1
    ls = 1

    its = 1
    time0 = time.time()
    for n in range(its):
        py_res = two_body(x1, x2, d1, d2, sig, ls)
    time1 = time.time()

    # try out numba kernel
    type_conv = {'C': 0, 'Si': 1}
    x1 = Two_Body_Env_Old(len(envs[0]['dists']))
    x2 = Two_Body_Env_Old(len(envs[1]['dists']))
    x1.dict_to_class(envs[0], type_conv)
    x2.dict_to_class(envs[1], type_conv)

    d1 = 0
    d2 = 1
    sig = 1
    ls = 1

    # compile for the first time
    jit_res = two_body_jit(x1.relative_coordinates,
                           x2.relative_coordinates,
                           x1.distances, x2.distances,
                           x1.central_atom, x2.central_atom,
                           x1.types, x2.types, d1, d2, sig, ls)

    time2 = time.time()
    for n in range(its):
        jit_res = two_body_jit(x1.relative_coordinates,
                               x2.relative_coordinates,
                               x1.distances, x2.distances,
                               x1.central_atom, x2.central_atom,
                               x1.types, x2.types, d1, d2, sig, ls)
    time3 = time.time()

    py_time = (time1-time0) / its
    jit_time = (time3 - time2) / its

    print(py_time)
    print(jit_time)
    print(py_time / jit_time)
