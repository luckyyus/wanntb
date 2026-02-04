from dataclasses import dataclass, field
from typing import List
import numpy as np




@dataclass
class SymmetryOperator:
    """Space group symmetry operation {R|t}."""
    rotation: np.ndarray
    translation: np.ndarray
    time_reversal: int = -1

    def __post_init__(self):
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        self.translation = np.asarray(self.translation, dtype=np.float64)

    def copy(self) -> 'SymmetryOperator':
        return SymmetryOperator(
            rotation=self.rotation.copy(),
            translation=self.translation.copy(),
            time_reversal=self.time_reversal
        )

    def inverse(self) -> 'SymmetryOperator':
        inv_rot = np.linalg.inv(self.rotation)
        inv_trans = -inv_rot @ self.translation
        return SymmetryOperator(rotation=inv_rot, translation=inv_trans, time_reversal=self.time_reversal)


@dataclass
class SmallGroup:
    elements: List[int] = field(default_factory=list)
    @property
    def order(self) -> int:
        return len(self.elements)


@dataclass
class ProjectionGroup:
    element: str
    orbitals: List[str] = field(default_factory=list)
    zaxis: str = "0,0,0"
    xaxis: str = "0,0,0"
    yaxis: str = "0,0,0"
    radial: str = "1"
    zona: str = "1.0"