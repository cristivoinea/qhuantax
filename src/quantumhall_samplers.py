from typing import Sequence, Tuple, Optional, Union
from jaxtyping import Key
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import quantax as qtx
from quspin.basis import spinless_fermion_basis_1d, spinful_fermion_basis_1d


def _get_lz_mask_from_basis_states(
    basis_states: np.ndarray,
    L: int,
    lz: int,
    nflav: int,
) -> np.ndarray:
    spins = qtx.utils.ints_to_array(np.asarray(basis_states), Nmodes=nflav * L)
    occ = (spins > 0).astype(np.int8)
    orbital_idx = np.arange(nflav * L) % L
    doubled_lz = occ @ (2 * orbital_idx - (L - 1))
    return doubled_lz == 2 * lz


def _get_squeezed_mask_from_basis_states(
    basis_states: np.ndarray,
    L: int,
    root_partition: np.ndarray,
) -> np.ndarray:
    if root_partition is None:
        raise ValueError("`root_partition` must be specified when `squeezed_basis=True`.")

    root_desc = np.sort(np.asarray(root_partition))[::-1]
    root_cumsum = np.cumsum(root_desc)
    n_particles = root_desc.size

    spins = qtx.utils.ints_to_array(np.asarray(basis_states), Nmodes=L)
    pos = np.where(spins > 0, np.arange(L), -1)
    lam_desc = np.sort(pos, axis=1)[:, ::-1][:, :n_particles]
    lam_cumsum = np.cumsum(lam_desc, axis=1)
    return np.all(lam_cumsum <= root_cumsum[None, :], axis=1)


def _get_proj_mask_from_basis_states(
    basis_states: np.ndarray,
    L: int,
    lz: int,
    nflav: int,
    squeezed_basis: bool = False,
    root_partition: Optional[np.ndarray] = None,
) -> np.ndarray:
    mask = _get_lz_mask_from_basis_states(basis_states, L, lz, nflav)

    if squeezed_basis:
        if nflav != 1:
            raise NotImplementedError(
                "Squeezed-basis projector masks are only implemented for spinless fermions."
            )
        mask &= _get_squeezed_mask_from_basis_states(basis_states, L, root_partition)

    return np.flatnonzero(mask)


def GetLzDenseProjector(L, N, lz, nflav=1, squeezed_basis = False, root_partition: Optional[np.ndarray] = None) -> np.array:
    if nflav == 1:
        basis = spinless_fermion_basis_1d(L, N).states

    elif nflav == 2:
        basis = spinful_fermion_basis_1d(L, zip(range(N,-1,-1), range(N+1)))
        basis = basis.states
    else:
        raise ValueError(f"Unsupported nflav={nflav}.")

    return _get_proj_mask_from_basis_states(
        basis, L, lz, nflav, squeezed_basis=squeezed_basis, root_partition=root_partition
    )


def GetLzSymmetryProjector(
    L: int,
    N: int,
    lz: int,
    symm: qtx.symmetry.Symmetry,
    nflav: int = 1,
    squeezed_basis: bool = False,
    root_partition: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return the indices of basis states in a symmetry-reduced QuSpin basis that belong
    to the requested Lz sector.

    This is the symmetry-aware counterpart of ``GetLzDenseProjector`` and is intended
    for dense vectors produced with the same ``symm`` via ``H.diagonalize(symm=...)`` or
    ``state.todense(symm=...)``.
    """
    del N  # The particle number is already encoded in `symm.basis`.
    symm.basis_make()
    return _get_proj_mask_from_basis_states(
        symm.basis.states,
        L,
        lz,
        nflav,
        squeezed_basis=squeezed_basis,
        root_partition=root_partition,
    )


def _propose_exchange_dipolecons(
    key: Key,
    old_spins: jax.Array,
    hopping_particle: jax.Array,
    neighbors: jax.Array,
    spinful: bool
) -> jax.Array:
    nsamples, Nmodes = old_spins.shape
    keys = jr.split(key, 3 * nsamples + spinful)

    arange = jnp.arange(nsamples)
    p_site = old_spins == hopping_particle
    #print("p_site_1 = ",p_site)
    choice_vmap = jax.vmap(lambda key, p: jr.choice(key, Nmodes, p=p))
    particle_idx1 = choice_vmap(keys[:nsamples], p_site)
    #print("particle_idx1 = ",particle_idx1)
    p_site = p_site.at[arange, particle_idx1].set(False)
    #print("p_site_2 = ",p_site)
    particle_idx2 = choice_vmap(keys[nsamples:2*nsamples], p_site)
    #print("particle_idx2 = ",particle_idx2)

    neighbors1 = neighbors[particle_idx1]
    choice_vmap = jax.vmap(lambda key, neighbor: jr.choice(key, neighbor))
    neighbor_idx1 = choice_vmap(keys[2*nsamples:3*nsamples], neighbors1)
    neighbor_idx1 = jnp.where(neighbor_idx1 == -1, particle_idx1, neighbor_idx1)
    #print(neighbor_idx1)

    if spinful:
        neighbor_idx2 = particle_idx1%(Nmodes//2) + particle_idx2%(Nmodes//2) - neighbor_idx1%(Nmodes//2)
        
        neighbor_idx1 = jnp.where((neighbor_idx2 < 0) | (neighbor_idx2 >= (Nmodes//2)), particle_idx1, neighbor_idx1)
        neighbor_idx2 = jnp.where((neighbor_idx2 < 0) | (neighbor_idx2 >= (Nmodes//2)), particle_idx2, neighbor_idx2)

        #print(particle_idx1%(Nmodes//2), particle_idx2%(Nmodes//2) , neighbor_idx1%(Nmodes//2), neighbor_idx2%(Nmodes//2))
        neighbor_spinflip = jr.randint(key, (nsamples,), 0, 2)
        neighbor_idx2 = neighbor_idx2 + neighbor_spinflip*(Nmodes//2)

        
        neighbor_idx1 = jnp.where(neighbor_idx2 == neighbor_idx1, particle_idx1, neighbor_idx1)
        neighbor_idx2 = jnp.where(neighbor_idx1 == particle_idx1, particle_idx2, neighbor_idx2)
        
        #print(particle_idx1, particle_idx2 , neighbor_idx1, neighbor_idx2)
    else:
        neighbor_idx2 = particle_idx1 + particle_idx2 - neighbor_idx1
        neighbor_idx1 = jnp.where(neighbor_idx2 == neighbor_idx1, particle_idx2, neighbor_idx1)
        neighbor_idx1 = jnp.where((neighbor_idx2 < 0) | (neighbor_idx2 >= Nmodes), particle_idx1, neighbor_idx1)
        neighbor_idx2 = jnp.where((neighbor_idx2 < 0) | (neighbor_idx2 >= Nmodes), particle_idx2, neighbor_idx2)

    #print(neighbor_idx2)
    neighbor_idx2 = jnp.where((old_spins[arange,neighbor_idx1] == 1) | (old_spins[arange,neighbor_idx2] == 1), particle_idx2, neighbor_idx2)
    neighbor_idx1 = jnp.where((old_spins[arange,neighbor_idx1] == 1) | (old_spins[arange,neighbor_idx2] == 1), particle_idx1, neighbor_idx1)


    particle1 = old_spins[arange, particle_idx1]
    neighbor1 = old_spins[arange, neighbor_idx1]
    particle2 = old_spins[arange, particle_idx2]
    neighbor2 = old_spins[arange, neighbor_idx2]
    new_spins = old_spins
    new_spins = new_spins.at[arange, particle_idx1].set(neighbor1)
    new_spins = new_spins.at[arange, neighbor_idx1].set(particle1)
    new_spins = new_spins.at[arange, particle_idx2].set(neighbor2)
    new_spins = new_spins.at[arange, neighbor_idx2].set(particle2)

    return new_spins


def _propose_exchange_dipolecons_squeezed(
    key: Key,
    old_spins: jax.Array,
    hopping_particle: jax.Array,
    neighbors: jax.Array,
    spinful: bool,
    root_partition: jax.Array
) -> jax.Array:
    nsamples, Nmodes = old_spins.shape
    keys = jr.split(key, 3 * nsamples + spinful)
    arange = jnp.arange(nsamples)

    p_site = old_spins == hopping_particle
    choice_vmap = jax.vmap(lambda key, p: jr.choice(key, Nmodes, p=p))

    particle_idx1 = choice_vmap(keys[:nsamples], p_site)
    p_site = p_site.at[arange, particle_idx1].set(False)
    particle_idx2 = choice_vmap(keys[nsamples:2*nsamples], p_site)

    neighbors1 = neighbors[particle_idx1]
    choice_vmap = jax.vmap(lambda key, neighbor: jr.choice(key, neighbor))
    neighbor_idx1 = choice_vmap(keys[2*nsamples:3*nsamples], neighbors1)
    neighbor_idx1 = jnp.where(neighbor_idx1 == -1, particle_idx1, neighbor_idx1)

    if spinful:
        neighbor_idx2 = particle_idx1%(Nmodes//2) + particle_idx2%(Nmodes//2) - neighbor_idx1%(Nmodes//2)
        
        neighbor_idx1 = jnp.where((neighbor_idx2 < 0) | (neighbor_idx2 >= (Nmodes//2)), particle_idx1, neighbor_idx1)
        neighbor_idx2 = jnp.where((neighbor_idx2 < 0) | (neighbor_idx2 >= (Nmodes//2)), particle_idx2, neighbor_idx2)

        neighbor_spinflip = jr.randint(key, (nsamples,), 0, 2)
        neighbor_idx2 = neighbor_idx2 + neighbor_spinflip*(Nmodes//2)

        
        neighbor_idx1 = jnp.where(neighbor_idx2 == neighbor_idx1, particle_idx1, neighbor_idx1)
        neighbor_idx2 = jnp.where(neighbor_idx1 == particle_idx1, particle_idx2, neighbor_idx2)
        
    else:
        neighbor_idx2 = particle_idx1 + particle_idx2 - neighbor_idx1
        neighbor_idx1 = jnp.where(neighbor_idx2 == neighbor_idx1, particle_idx2, neighbor_idx1)
        neighbor_idx1 = jnp.where((neighbor_idx2 < 0) | (neighbor_idx2 >= Nmodes), particle_idx1, neighbor_idx1)
        neighbor_idx2 = jnp.where((neighbor_idx2 < 0) | (neighbor_idx2 >= Nmodes), particle_idx2, neighbor_idx2)

    #print(neighbor_idx2)
    neighbor_idx2 = jnp.where((old_spins[arange,neighbor_idx1] == hopping_particle) | (old_spins[arange,neighbor_idx2] == hopping_particle), particle_idx2, neighbor_idx2)
    neighbor_idx1 = jnp.where((old_spins[arange,neighbor_idx1] == hopping_particle) | (old_spins[arange,neighbor_idx2] == hopping_particle), particle_idx1, neighbor_idx1)

    # 4. Build the tentative new state
    particle1 = old_spins[arange, particle_idx1]
    neighbor1 = old_spins[arange, neighbor_idx1]
    particle2 = old_spins[arange, particle_idx2]
    neighbor2 = old_spins[arange, neighbor_idx2]
    
    potential_spins = old_spins
    potential_spins = potential_spins.at[arange, particle_idx1].set(neighbor1)
    potential_spins = potential_spins.at[arange, neighbor_idx1].set(particle1)
    potential_spins = potential_spins.at[arange, particle_idx2].set(neighbor2)
    potential_spins = potential_spins.at[arange, neighbor_idx2].set(particle2)

    # 5. Evaluate Move Validity (Collisions & Squeezing)
    
    # A move is invalid if the target sites were already occupied by ANY particle (assuming hard-core)
    #is_occupied_1 = (old_spins[arange, neighbor_idx1] == hopping_particle) & (neighbor_idx1 != particle_idx2)
    #is_occupied_2 = (old_spins[arange, neighbor_idx2] == hopping_particle) & (neighbor_idx2 != particle_idx1)
    
    #collision = is_occupied_1 | is_occupied_2
    
    # Are we unsqueezing? (Distance increases)
    delta_init = jnp.abs(particle_idx1 - particle_idx2)
    delta_fin = jnp.abs(neighbor_idx1 - neighbor_idx2)
    is_unsqueezing = delta_fin > delta_init

    # Vectorized Dominance Check
    N_particles = root_partition.shape[0]
    
    # Extract positions of particles in the potential state, padding empty sites with -1
    pos = jnp.where(potential_spins == hopping_particle, jnp.arange(Nmodes), -1)
    
    # Sort descending. Valid positions group at the front, -1s at the back.
    sorted_pos = jnp.sort(pos, axis=1)[:, ::-1]
    real_pos = sorted_pos[:, :N_particles]

    #print(real_pos)
    
    # Prefix sums for dominance: Sum_new <= Sum_root
    new_cumsums = jnp.cumsum(real_pos, axis=1)
    root_cumsums = jnp.cumsum(jnp.sort(root_partition)[::-1])
    is_dominated = jnp.all(new_cumsums <= root_cumsums, axis=1)

    #print(new_cumsums)

    #print(root_cumsums)

    #print(is_dominated)

    # 6. Final Selection
    # Accept if: no collision AND (not squeezed basis OR not unsqueezing OR it passes dominance check)
    valid_move = ( ~is_unsqueezing | is_dominated) #& ~collision

    # Use a single jnp.where to filter all samples simultaneously
    new_spins = jnp.where(valid_move[:, None], potential_spins, old_spins)

    return new_spins


def _dominance_from_spins(
    spins: jax.Array, hopping_particle: int, root_partition: jax.Array
) -> jax.Array:
    nsamples, Nmodes = spins.shape
    n_particles = root_partition.shape[0]
    pos = jnp.where(spins == hopping_particle, jnp.arange(Nmodes), Nmodes)
    pos = jnp.sort(pos, axis=1)[:, :n_particles]
    pos_desc = pos[:, ::-1]
    root_desc = jnp.sort(root_partition)[::-1]
    root_cumsum = jnp.cumsum(root_desc)
    return jnp.all(jnp.cumsum(pos_desc, axis=1) <= root_cumsum[None, :], axis=1)


def _get_pair_data(
    spins: jax.Array,
    hopping_particle: int,
    root_partition: jax.Array,
    max_hop: int,
) -> Tuple[jax.Array, jax.Array]:
    Nmodes = spins.shape[0]
    n_particles = root_partition.shape[0]
    occ = jnp.where(spins == hopping_particle, jnp.arange(Nmodes), Nmodes)
    occ = jnp.sort(occ)[:n_particles]

    pair_i, pair_j = jnp.triu_indices(n_particles, k=1)
    a = occ[pair_i][:, None]
    b = occ[pair_j][:, None]

    c = jnp.arange(Nmodes)[None, :]
    d = a + b - c

    a_flat = jnp.broadcast_to(a, d.shape).reshape(-1)
    b_flat = jnp.broadcast_to(b, d.shape).reshape(-1)
    c_flat = jnp.broadcast_to(c, d.shape).reshape(-1)
    d_flat = d.reshape(-1)

    in_bounds = (c_flat >= 0) & (c_flat < Nmodes) & (d_flat >= 0) & (d_flat < Nmodes)
    distinct = c_flat < d_flat

    occ_c = spins[c_flat] == hopping_particle
    occ_d = spins[d_flat] == hopping_particle
    collision_c = occ_c & (c_flat != a_flat) & (c_flat != b_flat)
    collision_d = occ_d & (d_flat != a_flat) & (d_flat != b_flat)
    no_collision = ~(collision_c | collision_d)

    hop1 = jnp.maximum(jnp.abs(c_flat - a_flat), jnp.abs(d_flat - b_flat))
    hop2 = jnp.maximum(jnp.abs(c_flat - b_flat), jnp.abs(d_flat - a_flat))
    hop_ok = jnp.minimum(hop1, hop2) <= max_hop

    cand = jnp.broadcast_to(spins, (a_flat.shape[0], Nmodes))
    cand = cand.at[jnp.arange(a_flat.shape[0]), a_flat].set(-hopping_particle)
    cand = cand.at[jnp.arange(a_flat.shape[0]), b_flat].set(-hopping_particle)
    cand = cand.at[jnp.arange(a_flat.shape[0]), c_flat].set(hopping_particle)
    cand = cand.at[jnp.arange(a_flat.shape[0]), d_flat].set(hopping_particle)

    dominated = _dominance_from_spins(cand, hopping_particle, root_partition)
    changed = jnp.any(cand != spins[None, :], axis=1)
    valid = in_bounds & distinct & no_collision & hop_ok & dominated & changed
    return cand, valid


def _count_valid_pair_moves(
    spins: jax.Array,
    hopping_particle: int,
    root_partition: jax.Array,
    max_hop: int,
) -> jax.Array:
    _, valid = _get_pair_data(spins, hopping_particle, root_partition, max_hop)
    return jnp.sum(valid)


def _single_symmetric_squeezed_move(
    key: Key,
    spins: jax.Array,
    hopping_particle: int,
    root_partition: jax.Array,
    max_hop: int,
) -> Tuple[jax.Array, jax.Array]:
    cand, valid = _get_pair_data(spins, hopping_particle, root_partition, max_hop)
    n_valid_old = jnp.sum(valid)
    valid_idx = jnp.flatnonzero(valid, size=valid.shape[0], fill_value=0)
    choice = jr.randint(key, (), 0, jnp.maximum(n_valid_old, 1))
    picked = valid_idx[choice]
    proposed = cand[picked]
    has_move = n_valid_old > 0
    new_spins = jnp.where(has_move, proposed, spins)

    n_valid_new = _count_valid_pair_moves(
        new_spins, hopping_particle, root_partition, max_hop
    )
    denom = jnp.maximum(n_valid_new, 1)
    propose_ratio = jnp.where(has_move, n_valid_old / denom, 1.0)
    return new_spins, propose_ratio


@partial(jax.jit, static_argnums=4)
def _propose_exchange_dipolecons_squeezed_symmetric(
    key: Key,
    old_spins: jax.Array,
    hopping_particle: int,
    root_partition: jax.Array,
    max_hop: int,
) -> Tuple[jax.Array, jax.Array]:
    keys = jr.split(key, old_spins.shape[0])
    return jax.vmap(
        _single_symmetric_squeezed_move,
        in_axes=(0, 0, None, None, None),
    )(keys, old_spins, hopping_particle, root_partition, max_hop)


def _get_max_neighbor_distance(n_neighbor: Union[int, Sequence[int]]) -> int:
    neighbors = [n_neighbor] if isinstance(n_neighbor, int) else list(n_neighbor)
    return int(np.max(neighbors))



def _get_site_neighbors(n_neighbor: Union[int, Sequence[int]]) -> jax.Array:
    """
    Get the neighboring sites for each site.
    """
    sites = qtx.get_sites()
    n_neighbor = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
    neighbors = sites.get_neighbor(n_neighbor)
    neighbors = np.concatenate(neighbors, axis=0)
    neighbor_matrix = np.zeros((sites.Nsites, sites.Nsites), dtype=np.bool_)
    neighbor_matrix[neighbors[:, 0], neighbors[:, 1]] = True
    neighbor_matrix = neighbor_matrix | neighbor_matrix.T
    max_neighbors = np.max(np.sum(neighbor_matrix, axis=1)).item()
    neighbor_matrix = jnp.asarray(neighbor_matrix, dtype=jnp.bool_)
    fn = jax.vmap(lambda x: jnp.flatnonzero(x, size=max_neighbors, fill_value=-1))
    neighbors = fn(neighbor_matrix)
    neighbors = jnp.asarray(neighbors, dtype=jnp.int32, device=qtx.utils.get_replicate_sharding())

    if sites.particle_type == qtx.PARTICLE_TYPE.spinful_fermion:
        neighbors_spin = jnp.where(neighbors == -1, -1, neighbors + sites.Nsites)
        neighbors_u = jnp.hstack((neighbors, neighbors_spin, (jnp.arange(sites.Nsites) + sites.Nsites).reshape(sites.Nsites,1)))
        neighbors_d = jnp.hstack((neighbors, neighbors_spin, (jnp.arange(sites.Nsites)).reshape(sites.Nsites,1)))

        neighbors = jnp.concatenate([neighbors_u, neighbors_d], axis=0)
    return neighbors



class FermionTwoBodyDipoleCons(qtx.sampler.Metropolis):
    """
    Generate Monte Carlo samples by hopping random fermions to neighbor sites.
    This sampler only works when the system has fixed number of fermions.
    """

    def __init__(
        self,
        state: qtx.state.State,
        nsamples: int,
        reweight: float = 2.0,
        thermal_steps: Optional[int] = None,
        sweep_steps: Optional[int] = None,
        initial_spins: Optional[jax.Array] = None,
        n_neighbor: Union[int, Sequence[int]] = 1,
        squeezed_basis: bool = False,
        root_partition: Optional[jax.Array]= None
    ):
        r"""
        :param state:
            The state used for computing the wave function and probability.
            Since exchanging neighbor spins doesn't change the total Sz,
            the state must have `quantax.symmetry.ParticleConserve` symmetry to specify
            the symmetry sector.

        :param nsamples:
            Number of samples generated per iteration.
            It should be a multiple of the total number of machines to allow samples
            to be equally distributed on different machines.

        :param reweight:
            The reweight factor n defining the sample probability :math:`|\psi|^n`,
            default to 2.0.

        :param thermal_steps:
            The number of thermalization steps in the beginning of each Markov chain,
            default to be 20 * fock state length.

        :param sweep_steps:
            The number of steps for generating new samples,
            default to be 2 * fock state length.

        :param initial_spins:
            The initial spins for every Markov chain before the thermalization steps,
            default to be random spins.

        :param n_neighbor:
            The neighbors to be considered by particle hoppings, default to nearest neighbors.
        """
        sites = qtx.get_sites()
        if sites.Nparticles is None:
            raise ValueError(
                "The number of fermions should be specified in sites for `ParticleHop` sampler."
            )

        if 2 * sites.Ntotal <= state.Nmodes:
            self._hopping_particle = 1
        else:
            self._hopping_particle = -1

        self._neighbors = _get_site_neighbors(n_neighbor)
        self.squeezed_basis = squeezed_basis
        self.root_partition = root_partition

        super().__init__(
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @property
    def particle_type(self) -> Tuple[qtx.PARTICLE_TYPE, ...]:
        return (qtx.PARTICLE_TYPE.spinful_fermion, qtx.PARTICLE_TYPE.spinless_fermion)

    @property
    def nflips(self) -> int:
        return 4

    @partial(jax.jit, static_argnums=0)
    def propose(self, key: Key, old_spins: jax.Array) -> jax.Array:
        if self.squeezed_basis:
            return _propose_exchange_dipolecons_squeezed(
                key, old_spins, self._hopping_particle, self._neighbors, (self.state.symm._particle_type  == qtx.PARTICLE_TYPE.spinful_fermion), self.root_partition
            )
        else:
            return _propose_exchange_dipolecons(
                key, old_spins, self._hopping_particle, self._neighbors, (self.state.symm._particle_type  == qtx.PARTICLE_TYPE.spinful_fermion)
            )


class FermionTwoBodyDipoleConsSymmetricSqueezed(qtx.sampler.Metropolis):
    """
    Symmetric spinless dipole-conserving sampler restricted to the squeezed basis.
    The proposal is uniform over valid pair rearrangements, with a Hastings factor
    given by the ratio of valid move counts in the old and new states.
    """

    def __init__(
        self,
        state: qtx.state.State,
        nsamples: int,
        reweight: float = 2.0,
        thermal_steps: Optional[int] = None,
        sweep_steps: Optional[int] = None,
        initial_spins: Optional[jax.Array] = None,
        n_neighbor: Union[int, Sequence[int]] = 1,
        root_partition: Optional[jax.Array] = None,
    ):
        sites = qtx.get_sites()
        if sites.Nparticles is None:
            raise ValueError(
                "The number of fermions should be specified in sites for `ParticleHop` sampler."
            )
        if sites.particle_type != qtx.PARTICLE_TYPE.spinless_fermion:
            raise ValueError("This sampler currently supports only spinless fermions.")
        if root_partition is None:
            raise ValueError("`root_partition` must be specified for the squeezed sampler.")

        self._hopping_particle = 1 if 2 * sites.Ntotal <= state.Nmodes else -1
        self.root_partition = jnp.asarray(root_partition)
        self._max_hop = _get_max_neighbor_distance(n_neighbor)

        super().__init__(
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @property
    def particle_type(self) -> Tuple[qtx.PARTICLE_TYPE, ...]:
        return (qtx.PARTICLE_TYPE.spinless_fermion,)

    @property
    def nflips(self) -> int:
        return 4

    @partial(jax.jit, static_argnums=0)
    def propose(self, key: Key, old_spins: jax.Array) -> Tuple[jax.Array, jax.Array]:
        return _propose_exchange_dipolecons_squeezed_symmetric(
            key,
            old_spins,
            self._hopping_particle,
            self.root_partition,
            self._max_hop,
        )
