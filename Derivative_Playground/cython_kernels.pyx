cimport cython
from libc.math cimport exp

# get two body kernel between chemical environments
def two_body_c(dict x1,dict x2,str d1,str d2,float sig,float ls):
    cdef float d= sig*sig/(ls*ls*ls*ls)
    cdef float e= ls*ls
    cdef float f= 1/(2*ls*ls)
    cdef float kern = 0
    
    # record central atom types
    cdef str c1 = x1['central_atom']
    cdef str c2 = x2['central_atom']
    
    cdef int m, n
    
    cdef int x1_len = len(x1['dists'])
    cdef int x2_len = len(x2['dists'])
    
    cdef str e1, e2
    cdef float r1, coord1, r2, coord2, rr
    
    for m in range(x1_len):
        e1 = x1['types'][m]
        r1 = x1['dists'][m]
        coord1 = x1[d1][m]
        for n in range(x2_len):
            e2 = x2['types'][n]
            r2 = x2['dists'][n]
            coord2 = x2[d2][n]
            
            # check that atom types match
            if (c1==c2 and e1==e2) or (c1==e2 and c2==e1):
                rr = (r1-r2)*(r1-r2)
                kern += d*exp(-f*rr)*coord1*coord2*(e-rr)
                
    return kern

# get three body kernel between chemical environments
@cython.boundscheck(False)
def three_body_c(dict x1,dict x2,str d1,str d2,float sig,float ls):
    cdef float d= sig*sig/(ls*ls*ls*ls)
    cdef float e= ls*ls
    cdef float f= 1/(2*ls*ls)
    cdef float kern = 0
    
    cdef int m, n, p, q, typ
    
    cdef list x1_labs = x1['trip_dict']['labs']
    cdef list x2_labs = x2['trip_dict']['labs']
    cdef int x1_lab_len = len(x1_labs)
    cdef int x2_lab_len = len(x2_labs)
    cdef list x1_lab, x2_lab
    
    cdef list x1_typs = x1['trip_dict']['typs']
    
    cdef float ri1, ri2, ri3, ci1, ci2, ci3, rj1, rj2, rj3, cj1, cj2, cj3, k3
    
    cdef list x1_dists = x1['trip_dict']['dists']
    cdef list x2_dists = x2['trip_dict']['dists']
    cdef list x1_coords = x1['trip_dict'][d1]
    cdef list x2_coords = x2['trip_dict'][d2]
    
    cdef float rsum, r11, r12, r13, r21, r22, r23, r31, r32, r33, rr11, rr12, rr13, rr21, rr22, rr23, rr31, rr32,\
                rr33, cc11, cc12, cc13, cc21, cc22, cc23, cc31, cc32, cc33, rci11, rci12, rci13, rci21, rci22, rci23,\
                rci31, rci32, rci33, rcj11, rcj12, rcj13, rcj21, rcj22, rcj23, rcj31, rcj32, rcj33
    
    for m in range(x1_lab_len):
        x1_lab = x1_labs[m]

        for n in range(x2_lab_len):
            x2_lab = x2_labs[n]

            # check triplet type
            if x1_lab==x2_lab:
                # loop over tripets of the same type
                typ = x1_typs[m]

                # loop over triplets in environment 1
                for p in range(len(x1_dists[m])):
                    # set distances
                    ri1 = x1_dists[m][p][0]
                    ri2 = x1_dists[m][p][1]
                    ri3 = x1_dists[m][p][2]

                    # set coordinates
                    ci1 = x1_coords[m][p][0]
                    ci2 = x1_coords[m][p][1]
                    ci3 = x1_coords[m][p][2]

                    # loop over triplets in environment 2
                    for q in range(len(x2_dists[n])):
                        # set distances
                        rj1 = x2_dists[n][q][0]
                        rj2 = x2_dists[n][q][1]
                        rj3 = x2_dists[n][q][2]

                        # set coordinates
                        cj1 = x2_coords[n][q][0]
                        cj2 = x2_coords[n][q][1]
                        cj3 = x2_coords[n][q][2]

                        # add to kernel
                        if typ==1:
                            rsum = ri1*ri1+ri2*ri2+ri3*ri3+rj1*rj1+rj2*rj2+rj3*rj3
                            r11 = ri1-rj1
                            r12 = ri1-rj2
                            r13 = ri1-rj3
                            r21 = ri2-rj1
                            r22 = ri2-rj2
                            r23 = ri2-rj3
                            r31 = ri3-rj1
                            r32 = ri3-rj2
                            r33 = ri3-rj3

                            rr11 = ri1*rj1
                            rr12 = ri1*rj2
                            rr13 = ri1*rj3
                            rr21 = ri2*rj1
                            rr22 = ri2*rj2
                            rr23 = ri2*rj3
                            rr31 = ri3*rj1
                            rr32 = ri3*rj2
                            rr33 = ri3*rj3

                            cc11 = ci1*cj1
                            cc12 = ci1*cj2
                            cc13 = ci1*cj3
                            cc21 = ci2*cj1
                            cc22 = ci2*cj2
                            cc23 = ci2*cj3
                            cc31 = ci3*cj1
                            cc32 = ci3*cj2
                            cc33 = ci3*cj3

                            rci11 = r11*ci1
                            rci12 = r12*ci1
                            rci13 = r13*ci1
                            rci21 = r21*ci2
                            rci22 = r22*ci2
                            rci23 = r23*ci2
                            rci31 = r31*ci3
                            rci32 = r32*ci3
                            rci33 = r33*ci3

                            rcj11 = r11*cj1
                            rcj12 = r12*cj2
                            rcj13 = r13*cj3
                            rcj21 = r21*cj1
                            rcj22 = r22*cj2
                            rcj23 = r23*cj3
                            rcj31 = r31*cj1
                            rcj32 = r32*cj2
                            rcj33 = r33*cj3
                            
                            kern+=d*exp(-f*rsum)*\
                                ((exp(2*f*(rr11+rr22+rr33))*(e*(cc11+cc22+cc33)-(rci11+rci22+rci33)*(rcj11+rcj22+rcj33)))+
                                (exp(2*f*(rr11+rr23+rr32))*(e*(cc11+cc23+cc32)-(rci11+rci23+rci32)*(rcj11+rcj23+rcj32)))+
                                (exp(2*f*(rr12+rr21+rr33))*(e*(cc12+cc21+cc33)-(rci12+rci21+rci33)*(rcj12+rcj21+rcj33)))+
                                (exp(2*f*(rr12+rr23+rr31))*(e*(cc12+cc23+cc31)-(rci12+rci23+rci31)*(rcj12+rcj23+rcj31)))+
                                (exp(2*f*(rr13+rr21+rr32))*(e*(cc13+cc21+cc32)-(rci13+rci21+rci32)*(rcj13+rcj21+rcj32)))+
                                (exp(2*f*(rr13+rr22+rr31))*(e*(cc13+cc22+cc31)-(rci13+rci22+rci31)*(rcj13+rcj22+rcj31))))

                        if typ==2:
                            rsum = ri1*ri1+ri2*ri2+ri3*ri3+rj1*rj1+rj2*rj2+rj3*rj3
    
                            r11 = ri1-rj1
                            r12 = ri1-rj2
                            r21 = ri2-rj1
                            r22 = ri2-rj2

                            r33 = ri3-rj3
                            rr33 = ri3*rj3
                            cc33 = ci3*cj3
                            rci33 = r33*ci3
                            rcj33 = r33*cj3

                            # sum over permutations
                            kern+=d*exp(-f*rsum)*\
                                ((exp(2*f*(ri1*rj1+ri2*rj2+rr33))*(e*(ci1*cj1+ci2*cj2+cc33)-(r11*ci1+r22*ci2+rci33)*(r11*cj1+r22*cj2+rcj33)))+\
                                (exp(2*f*(ri1*rj2+ri2*rj1+rr33))*(e*(ci1*cj2+ci2*cj1+cc33)-(r12*ci1+r21*ci2+rci33)*(r12*cj2+r21*cj1+rcj33))))
                                
                        if typ==3:
                            r11 = ri1-rj1
                            r22 = ri2-rj2
                            r33 = ri3-rj3
                            kern+=(e*(ci1*cj1+ci2*cj2+ci3*cj3)-(r11*ci1+r22*ci2+r33*ci3)*(r11*cj1+r22*cj2+r33*cj3))*\
                                    d*exp(-f*(r11*r11+r22*r22+r33*r33))
                        
    return kern