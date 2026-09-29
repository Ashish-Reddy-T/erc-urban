"""
Per-site settings for the finals. `SITE=<name> python -m erc_urban.agent`
applies `env` (unless already set in the shell) and gives `notes` to the
driver model. Everything here was learned on the practice bots, 2026-09-22..25.
"""
from __future__ import annotations

COMMON = {
    "STAGE": "1", "NAV": "fan", "USE_MISSION": "1", "V_NOM": "0.28",
    "GYRO_AXIS": "z",                 # every competition bot is a Mini+ (gravity/yaw on z)
    "HEADING_K_COMPASS": "0",         # compasses seen: 80 deg off, 20 % short, broken
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
}

SITES: dict[str, dict] = {
    "unlp": {
        "env": {"FAN_TERRAIN": "offroad"},
        "notes": "Argentina, UNLP university campus (unseen). Level lawns with trees, paths, "
                 "confined inside the campus. Grass and paths are fine to drive on; go around "
                 "tree trunks, benches, buildings and people. Kerbs and steps are the hazard: never "
                 "climb one, find a ramp or a gap. Dry leaves and grass that the planner marks red "
                 "are usually drivable: use push.",
    },
    "panama": {
        "env": {"FAN_TERRAIN": "offroad"},
        "notes": "Panama (Colon) park, mission-3: 8 checkpoints, a longer route in the same "
                 "grass/tree area as before. Grass, dry leaves and leaf litter are fine to drive "
                 "on: when the planner marks them red or keeps escaping on them, push straight "
                 "through toward the checkpoint. Go around tree trunks, big roots, rocks and "
                 "bushes. The shortest line to a checkpoint often crosses leaf litter under trees: "
                 "that is expected. Prefer 'auto' when the way is open.",
    },
    "costarica": {
        "env": {"FAN_TERRAIN": "offroad", "HEADING_K_COMPASS": "0.04"},
        "notes": "Costa Rica: a hillside house garden. DANGER: the lawn south of the house "
                 "drops in a steep bank toward the road (the CP2 side). Do not go down it; if "
                 "the ground ahead falls away or the robot tilts, turn back. Stay on the lawn and "
                 "the paved driveway. Avoid flower beds, bushes, a plastic children's play slide, "
                 "stone walls and the drainage gutter along them. The steel sliding gate at the "
                 "driveway is closed: do not push against it. A checkpoint that is behind a "
                 "closed gate or down the bank can still be claimed from up to 15 m away: get "
                 "within 15 m by a safe route, never by the bank.",
    },
    "selebi": {
        "env": {"FAN_TERRAIN": "urban"},
        "notes": "Selebi-Phikwe, Botswana: town streets, dirt verges, gravel. Prefer the "
                 "road edge or sidewalk; gravel and packed dirt are fine; avoid kerbs and ditches.",
    },
    "mexico": {
        "env": {"FAN_TERRAIN": "offroad"},
        "notes": "Tepetlaoxtoc, Mexico: a narrow dirt lane between stacks of dried grass/hay on "
                 "both sides. Stay centred in the lane and follow it; never push into the hay. "
                 "Use short 'drive' steps with the candidate that keeps the lane centred.",
    },
    "machakos": {
        "env": {"FAN_TERRAIN": "offroad"},
        "notes": "Machakos, Kenya: campus with footpaths edged by concrete kerbs, raised planting "
                 "beds, gravel strips and grassy banks. Follow the mapped footpaths; do not try "
                 "to climb kerbs or banks.",
    },
    "wuhan": {
        "env": {"FAN_TERRAIN": "urban", "HEADING_K_COMPASS": "0.04"},
        "notes": "Wuhan campus: paved plazas, stone bollards, parked bicycles and scooters, "
                 "people. Go around bollards and parked vehicles.",
    },
}


def apply(site: str, environ) -> str:
    """Set this site's defaults in environ (shell values win). Returns the notes."""
    s = SITES.get(site.lower())
    if s is None:
        raise SystemExit(f"unknown SITE={site!r}; one of {', '.join(SITES)}")
    for k, v in {**COMMON, **s["env"]}.items():
        environ.setdefault(k, v)
    environ.setdefault("HEADING_STATE", f"runs/heading_state_{site.lower()}.json")
    return s["notes"]
