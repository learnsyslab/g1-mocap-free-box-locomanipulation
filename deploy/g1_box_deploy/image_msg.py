# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""sensor_msgs/msg/Image with the ROS2 IDL field layout (unitree_sdk2py does not ship it).

Own module WITHOUT ``from __future__ import annotations``: cyclonedds resolves the
field types from the real annotations, and postponed (string) annotations fail.
CDR is positional, so the field order must match ROS2's Image.idl exactly.
"""

from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types
from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Image_(idl.IdlStruct, typename="sensor_msgs.msg.dds_.Image_"):
    header: Header_
    height: types.uint32
    width: types.uint32
    encoding: str
    is_bigendian: types.uint8
    step: types.uint32
    data: types.sequence[types.uint8]
