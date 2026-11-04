"""Run a two-update, four-arm smoke check for the terrain-scan pilot."""
from pathlib import Path

from scripts import run_g1_extero_pilot as pilot


pilot.OUTPUT = Path("results/g1_extero_pilot_20261105_v4_smoke2")
pilot.PROTOCOL["training"]["pilot_iterations"] = 2


if __name__ == "__main__":
    pilot.main()
