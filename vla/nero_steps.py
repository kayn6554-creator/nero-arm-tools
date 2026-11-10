"""NERO step text, shared by training (JsonDataset) and the robot client.

Training and inference must build exactly the same instruction string:

    "<task> Step k/4: <step name>."

``k`` is the frame's ``step_index`` (1..4).  During collection the step key is
pressed at the END of the previous step, so ``step_index`` switches to k+1 at
that moment; at test time the operator presses "step done" at the same moment.
"""
from __future__ import annotations

STEPS = {
    "weigh": (
        "grasp the object",
        "move above the scale",
        "place the object on the scale",
        "move away and return",
    ),
    "temp": (
        "grasp the thermometer",
        "move above the container",
        "insert the thermometer into the water and measure",
        "put the thermometer back",
    ),
    "wind": (
        "grasp the anemometer",
        "approach the measuring point",
        "measure the wind speed",
        "put the anemometer back",
    ),
    # 11-09 新任务。键名要和数据集名 nero_<键>_v1 的前缀一致（task_key 按前缀找）
    "noise": (
        "grasp the noise meter",
        "move to the measuring point",
        "measure the noise",
        "put the noise meter back",
    ),
    "soil": (
        "grasp the soil probe",
        "move above the soil",
        "insert the probe into the soil and measure",
        "pull out the probe and put it back",
    ),
    "irwater": (
        "grasp the infrared thermometer",
        "move above the container",
        "aim at the water surface and measure",
        "put the infrared thermometer back",
    ),
    "irobj": (
        "grasp the infrared thermometer",
        "approach the object",
        "aim at the object and measure",
        "put the infrared thermometer back",
    ),
}

# Client task keys (nero_tasks.json) -> step table keys.
CLIENT_TASKS = {"weigh": "weigh", "task2": "temp", "task3": "wind",
                "noise": "noise", "soil": "soil", "irwater": "irwater", "irobj": "irobj"}


def task_key(dataset: str) -> str:
    """Map a dataset name such as ``nero_temp_v1`` or ``nero_weigh_fail_v1``."""
    for key in STEPS:
        if str(dataset).startswith(f"nero_{key}"):
            return key
    raise ValueError(f"no NERO step table for dataset {dataset!r}")


def step_text(key: str, step: int) -> str:
    names = STEPS[key]
    step = int(step)
    if not 1 <= step <= len(names):
        raise ValueError(f"step must be 1..{len(names)}, got {step}")
    return f"Step {step}/{len(names)}: {names[step - 1]}."


def step_prompt(task: str, key: str, step: int) -> str:
    """Full instruction sent by the client, e.g. for the OpenPI ``prompt`` field."""
    return f"{task.strip()} {step_text(key, step)}"


def step_text_for_frame(traj: dict, frame: int) -> str:
    source = traj.get("source") or {}
    steps, dataset = source.get("step_index"), source.get("dataset")
    if steps is None or dataset is None:
        raise ValueError(
            f"step_prompt needs source.step_index and source.dataset in {traj.get('time', '<unknown>')}"
        )
    return step_text(task_key(dataset), steps[frame])
