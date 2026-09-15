import pytest

from planning.arm_mapping import ARM_MAP, parser_arm_to_sim_arm


@pytest.mark.parametrize(
    "arm, expected",
    [
        ("A", "left_arm"),
        ("B", "right_arm"),
        ("a", "left_arm"),
        ("b", "right_arm"),
        ("  A  ", "left_arm"),
        ("\tB\n", "right_arm"),
    ],
)
def test_parser_arm_to_sim_arm_valid(arm, expected):
    assert parser_arm_to_sim_arm(arm) == expected


@pytest.mark.parametrize("arm", ["C", "", "   ", "AB", "left_arm", "right_arm", None])
def test_parser_arm_to_sim_arm_invalid(arm):
    with pytest.raises(ValueError):
        parser_arm_to_sim_arm(arm)


def test_arm_map_contents():
    assert ARM_MAP == {"A": "left_arm", "B": "right_arm"}