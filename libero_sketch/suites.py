"""Suite registry: short suite names -> LIBERO benchmark/dataset identifiers."""

from __future__ import annotations

# suite -> (benchmark_name, dataset_dir, download_group)
# libero_10 ships inside LIBERO's `libero_100` download bundle.  libero_90 ("90") never
# enters training: `all` leaves it out, and it is generated only on request, as a held-out
# validation set (scripts/10_export_hf.py --split-as validation).
SUITES = {
    "spatial": ("libero_spatial", "libero_spatial", "libero_spatial"),
    "object": ("libero_object", "libero_object", "libero_object"),
    "goal": ("libero_goal", "libero_goal", "libero_goal"),
    "long": ("libero_10", "libero_10", "libero_100"),
    "90": ("libero_90", "libero_90", "libero_100"),
}

ALL_SUITES = [s for s in SUITES if s != "90"]


def resolve_suites(suite_arg: str) -> list[str]:
    """'all' -> every suite; 'spatial,goal' -> subset; validates names."""
    if suite_arg.strip().lower() == "all":
        return list(ALL_SUITES)
    modes = [s.strip().lower() for s in suite_arg.split(",") if s.strip()]
    unknown = [m for m in modes if m not in SUITES]
    if unknown:
        raise ValueError(f"Unknown suite(s) {unknown}; choose from {ALL_SUITES} or 'all'")
    return modes


def benchmark_name(mode: str) -> str:
    return SUITES[mode][0]


def dataset_dir(mode: str) -> str:
    return SUITES[mode][1]


def download_group(mode: str) -> str:
    return SUITES[mode][2]


def add_suite_arg(parser):
    parser.add_argument(
        "--suite", default="all",
        help=f"comma-separated subset of {ALL_SUITES}, or 'all' (default)")
    return parser
