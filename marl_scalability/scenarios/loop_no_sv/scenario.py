import random
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

gen_scenario(
    t.Scenario(),
    output_dir=Path(__file__).parent,
)
