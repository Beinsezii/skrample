from collections.abc import Sequence

from .common import EmbeddedTableau, Tableau, TableauType
from .providers import (
    RK1,
    RK2,
    RK3,
    RK4,
    RKE2,
    RKE3,
    RKE5,
    RKZ,
    SSP,
    WSO,
    CustomTableau,
    RK2Custom,
    RK3Custom,
    RK4Custom,
    Shanks1965,
    TableauProvider,
)

BUILTIN_TABLEAUX: Sequence[TableauProvider[Tableau]] = [
    *RK1,
    *RK2,
    *RK3,
    *RK4,
    *RKZ,
    *SSP,
]
"All usable explicit runge-kutta methods"
BUILTIN_EMBEDDED_TABLEAU: Sequence[TableauProvider[EmbeddedTableau]] = [
    *RKE2,
    *RKE3,
    *RKE5,
]
"All usable embedded runge-kutta methods"

GRAVEYARD: Sequence[TableauProvider[TableauType]] = [
    *WSO,
    *Shanks1965,
]
"Unfortunate methods that ended up kinda sucking across all models"
