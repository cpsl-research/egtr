import math
from typing import TYPE_CHECKING, List, Tuple

import numpy as np


if TYPE_CHECKING:
    from avstack.environment.objects import ObjectState


def wrap_angle_mpi_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def heading_difference(obj1, obj2):
    return wrap_angle_mpi_pi(obj1.yaw - obj2.yaw)


def validate_relation_list(relations: List[Tuple]) -> bool:
    """Some relations must have counterparts present

    Relations in the form of (subject, object, predicate)
    """
    for relation in relations:
        cp = REL_DICT[relation[2]].counterpart
        if cp is not None:
            relation_cp = (relation[1], relation[0], cp.name)
            if relation_cp not in relations:
                raise RuntimeError(
                    f"Relation {relation} present and expected to "
                    f"have counterpart {relation_cp}, but not found"
                )

    return True


class RelationOperator:
    """Base class for relation operators

    Assumes objects are in a standard LiDAR coordinate system aligned
    with the optical center of the camera-of-interest on the ego agent:
        - X is forward/backward
        - Y is left/right
        - Z is up/down

    We need to use the LiDAR coordinate frame due to the definition
    of yaw angle that persists throughout the Attitude/Box3D classes.

    The logic below assumes we are only operating in the front-half plane
    of the coordinate system. To operate in the rear-half plane, transform
    the coordinate frame to face backwards. Ideally, we only operate within
    the visible angles subtended by the camera.

    Note: this does not mean we need a LiDAR sensor, per-se, just that the
    coordinate frame axes for 3D objects must follow the above convention.
    """

    @property
    def counterpart(self) -> "RelationOperator":
        raise NotImplementedError

    def __call__(self, obj1: "ObjectState", obj2: "ObjectState", **kwargs) -> bool:
        if obj1.position.x[0] < 0:
            raise ValueError("Object 1 is in the rear-half plane")
        if obj2.position.x[0] < 0:
            raise ValueError("Object 2 is in the rear-half plane")
        return self._evaluate(obj1, obj2, **kwargs)

    @staticmethod
    def _evaluate():
        raise NotImplementedError


class Following(RelationOperator):
    """(obj1) {is following} (obj2) in the obj2 frame sense"""

    name = "following"

    @property
    def counterpart(self) -> RelationOperator:
        return FollowedBy

    @staticmethod
    def _evaluate(
        obj1: "ObjectState",
        obj2: "ObjectState",
        distance_long_max: float = 15,
        distance_lat_thresh: float = 10,
        difference_heading_thresh: float = math.pi / 5,
    ) -> bool:

        # put object 1 in the reference frame of obj 1
        pos1_in_obj2 = obj1.position.change_reference(
            obj2.as_reference(), inplace=False
        )

        # condition 1: obj1 closely behind obj2, longitudinally
        c1 = -distance_long_max <= pos1_in_obj2[0] <= 0.0

        # condition 2: obj1 almost even with obj2, laterally
        c2 = abs(pos1_in_obj2[1]) <= distance_lat_thresh

        # condition 3: heading angle difference is small
        c3 = abs(heading_difference(obj1, obj2)) <= difference_heading_thresh

        return c1 & c2 & c3


class FollowedBy(RelationOperator):
    """(obj1) {is followed by} (obj2) in the obj1 frame sense"""

    name = "followed-by"

    @property
    def counterpart(self) -> RelationOperator:
        return Following

    @staticmethod
    def _evaluate(obj1: "ObjectState", obj2: "ObjectState", *args, **kwargs) -> bool:
        return Following._evaluate(obj2, obj1, *args, **kwargs)


class SideBy(RelationOperator):
    """(obj1) {is side by side} (obj2) in the obj2 frame sense

    This should be symmetric
    """

    name = "side_by"

    @property
    def counterpart(self):
        return SideBy

    @staticmethod
    def _evaluate(
        obj1: "ObjectState",
        obj2: "ObjectState",
        distance_long_thresh: float = 3,
        distance_lat_thresh: float = 4,
        difference_heading_thresh: float = math.pi / 5,
    ) -> bool:

        # put object 1 in the reference frame of obj 2
        pos1_in_obj2 = obj1.position.change_reference(
            obj2.as_reference(), inplace=False
        )

        # condition 1: longitudinal position difference is small
        c1 = abs(pos1_in_obj2.x[1]) <= distance_long_thresh

        # condition 2: lateral position difference is small
        c2 = abs(pos1_in_obj2.x[0]) <= distance_lat_thresh

        # condition 3: heading angle difference is small
        c3 = abs(heading_difference(obj1, obj2)) <= difference_heading_thresh

        return c1 & c2 & c3


class FrontOf(RelationOperator):
    """(obj1) {is in front of} (obj2) in the ego frame sense"""

    name = "front_of"

    @property
    def counterpart(self) -> RelationOperator:
        return Behind

    @staticmethod
    def _evaluate(
        obj1: "ObjectState",
        obj2: "ObjectState",
        distance_lat_thresh: float = 5,
        distance_long_min: float = 1.0,
    ) -> bool:

        # condition 1: lateral distance is moderate
        c1 = (obj1.position.x[1] - obj2.position.x[1]) <= distance_lat_thresh

        # condition 2: obj1 longitudinal distance less than obj2
        c2 = obj2.position.x[0] <= (obj1.position.x[0] + distance_long_min)

        return c1 & c2


class Behind(RelationOperator):
    """(obj1) {is behind} (obj2) in the ego frame sense"""

    name = "behind"

    @property
    def counterpart(self) -> RelationOperator:
        return FrontOf

    @staticmethod
    def _evaluate(obj1: "ObjectState", obj2: "ObjectState", *args, **kwargs) -> bool:
        return FrontOf._evaluate(obj2, obj1, *args, **kwargs)


class Near(RelationOperator):
    """(obj1) {is near to} (obj2) in any frame"""

    name = "near"

    @property
    def counterpart(self) -> RelationOperator:
        return Near

    @staticmethod
    def _evaluate(
        obj1: "ObjectState",
        obj2: "ObjectState",
        distance_threshold: float = 8,
    ) -> bool:
        return obj1.distance(obj2) <= distance_threshold


class Far(RelationOperator):
    """(obj1) {is far from} (obj2) in any frame"""

    name = "far"

    @property
    def counterpart(self) -> RelationOperator:
        return Far

    @staticmethod
    def _evaluate(
        obj1: "ObjectState",
        obj2: "ObjectState",
        distance_threshold: float = 30,
    ) -> bool:
        return obj1.distance(obj2) >= distance_threshold


class Occluding(RelationOperator):
    """(obj1) {is occluding} (obj2) in the ego frame"""

    name = "occluding"

    @property
    def counterpart(self) -> RelationOperator:
        return OccludedBy

    @staticmethod
    def _evaluate(
        obj1: "ObjectState",
        obj2: "ObjectState",
        vector_angle_difference_min: float = 2 * np.pi / 150,
    ) -> bool:
        """Only evaluating in BEV for simplicity

        Thus, any vertical offsets are not taken into account.
        """
        # condition 1: obj1 is closer to ego than obj 2
        c1 = obj1.position.norm() < obj2.position.norm()

        # evaluate the rest if c1 is true
        if c1:
            # get azimuth angles to box corners
            corners1 = obj1.box.corners
            az1 = np.arctan2(corners1[:, 1], corners1[:, 0])
            corners2 = obj2.box.corners
            az2 = np.arctan2(corners2[:, 1], corners2[:, 0])

            # get left and right azimuths
            o1l, o1r = min(az1), max(az1)
            o2l, o2r = min(az2), max(az2)

            # condition 2: obj2 far left vector left of obj1 far right vector
            c2 = (vector_angle_difference_min + o2l) < o1r

            # condition 3: obj2 far right vector right of obj1 far left vector
            c3 = o2r < (o1l + vector_angle_difference_min)

            return c2 & c3
        else:
            return False


class OccludedBy(RelationOperator):
    """(obj1) {is occluded by} (obj2) in the ego frame"""

    name = "occluded_by"

    @property
    def counterpart(self) -> RelationOperator:
        return Occluding

    @staticmethod
    def _evaluate(obj1: "ObjectState", obj2: "ObjectState", *args, **kwargs) -> bool:
        return Occluding._evaluate(obj2, obj1, *args, **kwargs)


class NoRelation(RelationOperator):
    """No relation between obj1 and obj2"""

    name = "__background__"  # this is what EGTR called it...

    @property
    def counterpart(self) -> RelationOperator:
        return NoRelation

    @staticmethod
    def _evaluate(obj1: "ObjectState", obj2: "ObjectState", *args, **kwargs) -> bool:
        return False  # not sure if there is ever no relation...


RELATIONS = [
    NoRelation(),
    Following(),
    FollowedBy(),
    SideBy(),
    FrontOf(),
    Behind(),
    Near(),
    Far(),
    Occluding(),
    OccludedBy(),
]


REL_STRINGS = [REL.name for REL in RELATIONS]
REL_DICT = {REL.name: REL for REL in RELATIONS}
REL_INDEX = {i: REL.name for i, REL in enumerate(RELATIONS)}
REL_REVINDEX = {REL.name: i for i, REL in enumerate(RELATIONS)}
