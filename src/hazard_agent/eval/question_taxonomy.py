"""The 5-type spatial question taxonomy used for the ablation benchmark.

See docs/PROJECT_PLAN.md section 4 for the full description and why
each type exists. Out-of-view recall and change/hazard detection are
the headline results; in-view relational is the control.
"""

from enum import Enum


class QuestionType(str, Enum):
    IN_VIEW_RELATIONAL = "in_view_relational"
    OUT_OF_VIEW_RECALL = "out_of_view_recall"
    REFERENCE_FRAME = "reference_frame"
    INSTANCE_PERSISTENCE = "instance_persistence"
    CHANGE_HAZARD_DETECTION = "change_hazard_detection"
