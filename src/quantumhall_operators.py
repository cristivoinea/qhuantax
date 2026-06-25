from typing import Sequence
import numpy as np
import sympy.physics.wigner as spw
import quantax as qtx


def get_int_matrix(nm : int, ps_pot : Sequence[int]):
    """
        GetIntMatrix(nm :: Int64, ps_pot :: Vector{<:Number}) :: Array{ComplexF64, 3}

    Gives the interaction matrix ``U_{m_1,m_2,m_3,m_4}`` from the pseudopotentials.
        
    # Argument

    * `nm :: Int64` is the number of orbitals.
    * `ps_pot :: Vector{<:Number}` is the vector of non-zero pseudopotentials.

    # Output
    * A `nm`×`nm`×`nm` array giving the interaction matrix ``U_{m_1,m_2,m_3,-m_1-m_2-m_3}``.
    """
    int_el = np.zeros((nm, nm, nm))
    s = 0.5 * (nm - 1)
    for m1 in range(nm): 
        m1r = m1 - s
        for m2 in range(nm): 
            m2r = m2 - s
            for m3 in range(nm): 
                m3r = m3 - s
                m4 = m1 + m2 - m3 
                if m4 < 0 or m4 >= nm:
                    continue
                m4r = m4 - s
                for l in range(len(ps_pot)):
                    if (abs(m1r + m2r) > nm - l - 1) or (abs(m3r + m4r) > nm - l - 1):
                        break
                    int_el[m1, m2, m3] += ps_pot[l] * (2 * nm - 2 * l - 1) * \
                        spw.wigner_3j(s, s, nm - l - 1, m1r, m2r, -m1r - m2r) * \
                            spw.wigner_3j(s, s, nm - l - 1, m4r, m3r, -m3r - m4r)
    return int_el



def GetSpinlessDenIntTerms(nm : int, ps_pot : Sequence[int]) -> qtx.operator.Operator:
    """
        GetDenIntTerms(nm :: Int64, nf :: Int64[, ps_pot :: Vector{<:Number}][, mat_a :: Matrix{<:Number}[, mat_b :: Matrix{<:Number}]][ ; m_kept :: Vector{Int64}]) :: Terms

    Return the normal-ordered density-density term in the Hamiltonian 
    ```math 
    ∑_{\\{m_i,f_i\\}}U_{m_1m_2m_3m_4}M^A_{f_1f_4}M^B_{f_2f_3}c^{†}_{m_1f_1}c^{†}_{m_2f_2}c_{m_3f_3}c_{m_4f_4}.
    ```

    # Arguments 

    * `nm :: Int64` is the number of orbitals.
    * `ps_pot :: Vector{<:Number}` is a list of numbers specifying the pseudopotentials for the interacting matrix ``U_{m_1m_2m_3m_4}``. Facultative, `[1.0]` by default. 
    """
    H = 0
    interaction = get_int_matrix(nm, ps_pot)
    
    for m1 in range(nm):
        for m2 in range(nm):
            if m1 == m2:
                continue
            for m3 in range(nm):
                m4 = m1 + m2 - m3 
                if m4 < 0 or m4 >= nm or m3 == m4: 
                    continue
                val = interaction[m1, m2, m3]
                if (abs(val) < 1E-12):
                    continue  
                H += val * (qtx.operator.create(m1) @ qtx.operator.create(m2) @ qtx.operator.annihilate(m3) @ qtx.operator.annihilate(m4))
    return H

def qtx_create(f,m):
    if f == 0:
        return qtx.operator.create_u(m)
    else:
        return qtx.operator.create_d(m)
    

def qtx_annihilate(f,m):
    if f == 0:
        return qtx.operator.annihilate_u(m)
    else:
        return qtx.operator.annihilate_d(m)

def GetSpinfulDenIntTerms(nm : int, ps_pot : Sequence[int], mat_a : Sequence[int], mat_b : Sequence[int]) -> qtx.operator.Operator:
    """
        GetDenIntTerms(nm :: Int64, nf :: Int64[, ps_pot :: Vector{<:Number}][, mat_a :: Matrix{<:Number}[, mat_b :: Matrix{<:Number}]][ ; m_kept :: Vector{Int64}]) :: Terms

    Return the normal-ordered density-density term in the Hamiltonian 
    ```math 
    ∑_{\\{m_i,f_i\\}}U_{m_1m_2m_3m_4}M^A_{f_1f_4}M^B_{f_2f_3}c^{†}_{m_1f_1}c^{†}_{m_2f_2}c_{m_3f_3}c_{m_4f_4}.
    ```

    # Arguments 

    * `nm :: Int64` is the number of orbitals.
    * `ps_pot :: Vector{<:Number}` is a list of numbers specifying the pseudopotentials for the interacting matrix ``U_{m_1m_2m_3m_4}``. Facultative, `[1.0]` by default. 
    """
    H = 0
    interaction = get_int_matrix(nm, ps_pot)

    for o1 in range(2*nm):
        m1 = o1//2
        f1 = o1%2
        for o2 in range(2*nm):
            m2 = o2//2
            f2 = o2%2
            if o1 == o2:
                continue
            for o3 in range(2*nm):
                m3 = o3//2
                f3 = o3%2
                if np.abs(mat_b[f2, f3]) < 1e-13:
                     continue
                m4 = m1 + m2 - m3 
                if m4 < 0 or m4 >= nm: 
                    continue
                for f4 in range(2):
                    if np.abs(mat_a[f1, f4]) < 1e-13:
                        continue
                    o4 = m4 * 2 + f4
                    #print(o1, o2, o3, o4)
                    if o3 == o4:
                        continue
                    val = mat_a[f1, f4] * mat_b[f2, f3] * interaction[m1, m2, m3]
                    #print(val)
                    if (np.abs(val) < 1E-15):
                        continue
                    H += val * (qtx_create(f1, m1) @ qtx_create(f2, m2) @ qtx_annihilate(f3, m3) @ qtx_annihilate(f4, m4))
    return H


def GetSpinfulPolTerms(nm : int, mat : Sequence[int]) -> qtx.operator.Operator:
    H = 0
    for o1 in range(2*nm):
        m1 = o1//2
        f1 = o1%2
        for f2 in range(2):
            if np.abs(mat[f1, f2]) < 1E-13:
                continue
            H += mat[f1,f2] * (qtx_create(f1, m1) @ qtx_annihilate(f2, m1))

    return H


def GetSpinfulMultipoleTerms(
    nm: int,
    ell: int,
    m: int,
    mat: Sequence[int],
    normalize: bool = False,
) -> qtx.operator.Operator:
    r"""
    Return the spinful one-body multipole operator

        sum_{m1,m2,f1,f2} <S,m1|T_{ell,m}|S,m2> M_{f1,f2}
            c^\dagger_{m1,f1} c_{m2,f2}.

    The orbital matrix elements are taken from the standard rank-``ell`` tensor
    structure on the sphere. For variational warm starts, the overall normalization is
    usually irrelevant; only the relative coefficients matter.
    """
    if abs(m) > ell:
        raise ValueError(f"Need |m| <= ell, got ell={ell}, m={m}.")

    H = 0
    s = 0.5 * (nm - 1)
    prefactor = np.sqrt(2 * ell + 1)
    if normalize:
        prefactor /= np.sqrt(nm)

    for m1 in range(nm):
        m1r = m1 - s
        for m2 in range(nm):
            m2r = m2 - s
            coeff = prefactor * ((-1) ** int(round(s - m1r))) * spw.wigner_3j(
                s, ell, s, -m1r, m, m2r
            )
            coeff = float(coeff)
            if abs(coeff) < 1e-13:
                continue

            for f1 in range(2):
                for f2 in range(2):
                    if abs(mat[f1, f2]) < 1e-13:
                        continue
                    # Emit true density operators when the term is diagonal so Quantax
                    # can keep it in the cheap `apply_diag` path.
                    if m1 == m2 and f1 == f2:
                        if f1 == 0:
                            H += coeff * mat[f1, f2] * qtx.operator.number_u(m1)
                        else:
                            H += coeff * mat[f1, f2] * qtx.operator.number_d(m1)
                    else:
                        H += coeff * mat[f1, f2] * (
                            qtx_create(f1, m1) @ qtx_annihilate(f2, m2)
                        )
    return H


def GetGlobalSpinFlip(nm : int) -> qtx.operator.Operator:
    H = (qtx_create(0, 0) @ qtx_annihilate(1, 0) + qtx_create(1, 0) @ qtx_annihilate(0, 0))
    for m1 in range(1,nm):
        H = H @ (qtx_create(0, m1) @ qtx_annihilate(1, m1) + qtx_create(1, m1) @ qtx_annihilate(0, m1))
    return H 


def GetLzTerms(nm : int, nflav: int) -> qtx.operator.Operator:
    Lz = 0
    s = (nm - 1) / 2
    for o in range(nm * nflav):
        m = o//nflav
        f = o%nflav
        if nflav == 1:
            Lz += (m-s) * qtx.operator.number(m)
        else:
            Lz += (m-s) * (qtx_create(f, m) @ qtx_annihilate(f, m))

    return Lz


def GetLpTerms(nm : int, nflav: int) -> qtx.operator.Operator:
    Lp = 0
    for o in range(nflav, nm * nflav):
        m = o//nflav
        f = o%nflav
        if nflav == 1:
            Lp += np.sqrt(m * (nm - m)) * (qtx.operator.create(m) @ qtx.operator.annihilate(m-1))
        else:
            Lp += np.sqrt(m * (nm - m)) * (qtx_create(f, m) @ qtx_annihilate(f, m-1))

    return Lp


def GetL2Terms(nm : int, nflav: int) -> qtx.operator.Operator:
    Lz = GetLzTerms(nm, nflav)
    Lp = GetLpTerms(nm, nflav)
    Lm = Lp.H

    return Lz @ Lz - Lz + Lp @ Lm


def GetIdTerms(nm : int, nflav: int) -> qtx.operator.Operator:
    Op = 0
    for o in range(nm * nflav):
        m = o//nflav
        f = o%2
        if nflav == 1:
            Op += qtx.operator.number(m)
        else:
            Op += (qtx_create(f, m) @ qtx_annihilate(f, m))

    return Op
