"""Synthetic twin of the dual-scanner rig.

A DXF cross-section is extruded to a prism, two virtual scanners raycast it
from the rig's calibrated positions, and the PLYs are read back by the same
loader, projection and detector code. Pose, occlusion and crossing are set
directly.

Modules:
  `section.py`    DXF loops -> triangulated cross-section -> extruded prism
  `sensor.py`     pinhole range sensor shaped by the rig's measured geometry
  `generator.py`  6-DOF bar poses, bundle layouts, PLY + truth.json writing
  `evaluate.py`   scoring against exact truth (detection, localization, pairing)
"""

from .generator import (
    APPARENT_LENGTH_MM,
    BarPose,
    SyntheticScene,
    bundle_row,
    face_truth,
    load_truth,
    render_scene,
    scene_mesh,
)
from .section import SectionMesh, build_section_mesh, extrude_prism, section_mesh
from .sensor import SensorModel, raycast, sensor_for_side

__all__ = [
    "APPARENT_LENGTH_MM", "BarPose", "SyntheticScene", "bundle_row",
    "face_truth", "load_truth", "render_scene", "scene_mesh",
    "SectionMesh", "build_section_mesh", "extrude_prism", "section_mesh",
    "SensorModel", "raycast", "sensor_for_side",
]
