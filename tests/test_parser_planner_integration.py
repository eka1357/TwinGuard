from planning.command_planner import prepare_commands
from planning.language_parser import parse_instruction


def test_parser_to_planner_pipeline():
    parsed = parse_instruction(
        "Arm A, pick up the plate, then place it on the table."
    )

    prepared = prepare_commands(parsed)

    assert len(prepared) == 2

    assert prepared[0]["arm"] == "A"
    assert prepared[0]["sim_arm"] == "left_arm"
    assert prepared[0]["action"] == "pick"
    assert prepared[0]["obj"] == "plate"

    assert prepared[1]["arm"] == "A"
    assert prepared[1]["sim_arm"] == "left_arm"
    assert prepared[1]["action"] == "place"
    assert prepared[1]["obj"] == "plate"
    assert prepared[1]["target"] == "table"