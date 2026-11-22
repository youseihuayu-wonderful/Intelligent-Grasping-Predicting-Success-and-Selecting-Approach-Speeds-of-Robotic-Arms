"""Object definitions for the supervised grasp study, separate from storage tasks.

New assets are rigid geometric approximations, not scanned objects or material
models. Mass levels are prescribed experimental settings, not measurements.
"""

from dataclasses import dataclass
from types import MappingProxyType

from scenarios import GraspTarget, TARGETS


SCALE_LEVELS = (0.8, 1.0, 1.2)
FRICTION_LEVELS = (0.20, 0.40, 0.80, 1.20)


@dataclass(frozen=True)
class StudyObject:
    grasp_target: GraspTarget
    asset_path: str
    mass_levels_kg: tuple[float, ...]
    geometry_description: str


STUDY_OBJECTS = MappingProxyType({
    "bowl": StudyObject(
        TARGETS["bowl"],
        "vendor/libero/stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml",
        (0.06, 0.18, 0.30, 0.45),
        "Existing LIBERO black bowl; near-side rim pinch.",
    ),
    "bottle": StudyObject(
        TARGETS["bottle"],
        "vendor/libero/turbosquid_objects/wine_bottle/wine_bottle.xml",
        (0.06, 0.18, 0.30, 0.45),
        "Existing LIBERO wine bottle; neck pinch.",
    ),
    "spoon": StudyObject(
        GraspTarget("spoon", "Spoon", (0.0525, 0.0, 0.013),
                    "Top-down pinch across the raised spoon head; rigid geometric approximation."),
        "assets/grasp_study/spoon.xml",
        (0.015, 0.025, 0.040, 0.060),
        "150 x 36 x 17.5 mm rigid spoon with a tabletop handle and raised open concave head.",
    ),
    "glass": StudyObject(
        GraspTarget("glass", "Glass tumbler", (0.0, 0.0, 0.065),
                    "Top-down opposing-finger pinch around the upper cup wall."),
        "assets/grasp_study/glass.xml",
        (0.080, 0.150, 0.250, 0.350),
        "64 mm outer diameter x 90 mm high rigid hollow tumbler; 3 mm segmented walls and 4 mm bottom; no fracture or fluid model.",
    ),
    "pen": StudyObject(
        GraspTarget("pen", "Pen", (0.0, 0.0, 0.010),
                    "Top-down pinch across a horizontal pen barrel; no adhesion or attachment."),
        "assets/grasp_study/pen.xml",
        (0.005, 0.010, 0.020, 0.030),
        "140 x 12 x 12 mm rigid horizontal pen barrel with a tip and cap; no writing or bending model.",
    ),
})
