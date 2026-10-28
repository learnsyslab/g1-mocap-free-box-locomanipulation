# SBTO's G1 model (dummy hands)

`g1_mjx_dummy_hands.xml` is SBTO's own MuJoCo model of the 29-DoF G1 with dummy box
hands ([SBTO](https://github.com/Atarilab/sbto), `sbto/models/unitree_g1/`): its
actuator gains, primitive collision geometry and contact conventions are what SBTO's
trajectories are optimised against. `training/scripts/motion/add_computed_contact.py`
measures the reference's hand / box / floor contact schedule with this geometry.

Meshes come from `../meshes/g1/` (shared with the sim2sim model); the four that differ
or are missing there are in `meshes/`. Unitree's G1 description, BSD-3-Clause (see
THIRD_PARTY_NOTICES.md).
