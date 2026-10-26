# Third-party notices

This repository is licensed under Apache-2.0 (see `LICENSE`), except for the third-party
material below, which keeps its own license. Files adapted from third-party code say so
in their header.

## Redistributed in this repository

### Unitree Robotics: G1 robot model

* `assets/mujoco/g1_29dof_dummyhands.xml`: derived from `g1_29dof.xml` in
  [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) (modified: box-shaped
  "dummy hands", head camera, collision geometry).
* `assets/mujoco/meshes/g1/*.STL`: unmodified from unitree_mujoco.
* `assets/isaaclab/g1-29dof/*.usd`: the G1 29-DoF USD from Unitree's
  [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model) dataset,
  unmodified. That dataset declares no license; the USD is Unitree's conversion of the G1
  description in [unitree_ros](https://github.com/unitreerobotics/unitree_ros), which is
  BSD-3-Clause (Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD.).
  `g1_29dof_rev_1_0_primitive_collision.usda` is our override layer on top of it.

```
BSD 3-Clause License

Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Isaac Lab

* `training/scripts/rsl_rl/{train,play,cli_args}.py` and the extension scaffolding
  (`setup.py`, package `__init__.py` files, `agents/rsl_rl_ppo_cfg.py`): from the
  [Isaac Lab](https://github.com/isaac-sim/IsaacLab) extension template, modified.

```
Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).

All rights reserved.

SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### unitree_rl_lab

* `training/.../tracking/tracking_env_cfg.py` and `mdp/{commands,events,observations,rewards,terminations}.py`:
  adapted from the `tasks/mimic` motion-tracking task of
  [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab), modified (object
  tracking, box curriculum, depth student).
* Copyright 2024 The Isaac Lab Project Developers. Licensed under the Apache License,
  Version 2.0; the full text is in `LICENSE`.

### whole_body_tracking (BeyondMimic)

* The unitree_rl_lab `tasks/mimic` code above is itself based on
  [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking).

```
MIT License

Copyright (c) 2024, The Isaac Lab Project Developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### rsl_rl

* `training/.../tracking/distill_ppo.py`, `distill_algorithm.py`: parts of the PPO and
  distillation updates are adapted from [rsl_rl](https://github.com/leggedrobotics/rsl_rl).

```
Copyright (c) 2026, ETH Zurich
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

See licenses/dependencies for license information of dependencies of this package.
```

### mjlab-distill (Atarilab)

* `training/.../tracking/distill_models.py`, `distill_algorithm.py`, `distill_ppo.py`:
  adapted from Atarilab's mjlab-distill (Majid Khadiv, Shafeef Omar), modified for Isaac Lab.
* Copyright 2025, The mjlab Developers. Licensed under the Apache License, Version 2.0;
  the full text is in `LICENSE`.

### SBTO (Atarilab)

* The reference motion (`training/.../motions/sub3_realbox_dummy_hands.npz`,
  `deploy/policy/motion.npz`) was refined with SBTO, and the robot's PD gains are those of
  SBTO's MuJoCo model. SBTO is declared MIT-licensed.

### OmniRetarget dataset and OMOMO

* The reference motion starts from `robot-object/sub3_largebox_005` of the
  [OmniRetarget dataset](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset)
  (MIT), which is retargeted from the [OMOMO](https://github.com/lijiaman/omomo_release)
  human-object motion data (MIT).
* OMOMO's human motions are expressed with the SMPL family of body models, which have their
  own license terms (non-commercial research use). No SMPL model is included here.

## Git submodules (referenced, not copied)

These are pinned by commit and fetched from their own repositories by
`git submodule update --init`; each keeps its own license file.

* `training/third_party/IsaacLab`: [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
  v3.0.0-beta2.patch1, BSD-3-Clause (`source/isaaclab_mimic` is Apache-2.0).
* `deploy/third_party/unitree_sdk2_python`:
  [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python),
  BSD-3-Clause (Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD.).
* `deploy/third_party/Fast-FoundationStereo`: the
  [learnsyslab fork](https://github.com/learnsyslab/Fast-FoundationStereo) of NVIDIA's
  [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo), with pixi and
  TensorRT build tasks added. NVIDIA source code license: **non-commercial / research use
  only**. No weights or engines are included; the research weights are downloaded from
  NVIDIA's link in that repository.

## Not redistributed (installed by pixi)

NVIDIA Isaac Sim (NVIDIA proprietary license), NVIDIA TensorRT (NVIDIA proprietary license),
rsl_rl (BSD-3-Clause), PyTorch (BSD-3-Clause), MuJoCo (Apache-2.0), CycloneDDS (EPL-2.0 or
BSD-3-Clause), pygame (LGPL-2.1), and librealsense (Apache-2.0, built from source by
`deploy/scripts/build_pyrealsense.sh`).
