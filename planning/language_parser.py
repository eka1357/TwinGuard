import re
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class ParsedCommand:
    arm: Optional[str]      # "A" or "B"
    action: Optional[str]   # "pick", "place", "pour", "open"
    obj: Optional[str]      # "plate", "mug", "drawer"
    target: Optional[str]   # "table", "handoff", other arm's object, etc.

ARM_PATTERN = re.compile(r"\barm\s*([ab])\b", re.IGNORECASE)
ACTION_KEYWORDS = {
    "pick": ["pick up", "pick", "grab", "retrieve"],
    "place": ["place", "put down", "set down"],
    "pour": ["pour"],
    "open": ["open"],
}
OBJECT_KEYWORDS = ["plate", "mug", "drawer"]

def parse_segment(segment: str) -> ParsedCommand:
    seg = segment.lower()

    arm_match = ARM_PATTERN.search(seg)
    arm = arm_match.group(1).upper() if arm_match else None

    action = None
    for act, keywords in ACTION_KEYWORDS.items():
        if any(kw in seg for kw in keywords):
            action = act
            break

    obj = next((o for o in OBJECT_KEYWORDS if o in seg), None)

    target = "table" if "table" in seg else ("handoff" if "hand" in seg else None)

    return ParsedCommand(arm=arm, action=action, obj=obj, target=target)

def parse_instruction(instruction: str) -> list[dict]:
    segments = re.split(r",|\bthen\b", instruction)
    results = []
    last_arm = None
    last_obj = None
    for s in segments:
        if not s.strip():
            continue
        cmd = parse_segment(s)
        if cmd.arm is None:
            cmd.arm = last_arm
        else:
            last_arm = cmd.arm
        if cmd.obj is None:
            cmd.obj = last_obj
        else:
            last_obj = cmd.obj
        if cmd.action is None and cmd.target == "handoff":
            cmd.action = "handoff"
        if cmd.action is None:
            continue  # drop non-actionable fragments like "Arm A" alone
        results.append(asdict(cmd))
    return results


if __name__ == "__main__":
    
    test_cases = [
    "Arm A, pick up the plate.",                              # arm before action, no "with"
    "Pick up the mug with arm B then place it on the table.",  # "then" instead of comma
    "arm b pick up the plate",                                 # lowercase, no punctuation
    "Open the drawer with arm A, retrieve the plate.",         # drawer explicitly tied to an arm
    "Hand the plate to arm B.",                                # handoff phrasing, no "pick up" verb
    ]
    for t in test_cases:
        print(t)
        for cmd in parse_instruction(t):
            print(" ", cmd)
        print()