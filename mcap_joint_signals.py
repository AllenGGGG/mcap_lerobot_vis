"""Decode the named position feedback/commands recorded by ros2_control."""
from collections import defaultdict


JOINT_TOPIC = "/joint_states"
STATISTICS_ROOT = "/controller_manager/introspection_data"
COMMAND_TOPICS = (f"{STATISTICS_ROOT}/full", f"{STATISTICS_ROOT}/values")
NAMES_TOPIC = f"{STATISTICS_ROOT}/names"
FIXED_JOINTS = {"body_joint1", "body_joint2", "head_joint1", "head_joint2"}


def joint_positions(message):
    if len(message.name) != len(message.position):
        raise ValueError("JointState name/position 长度不一致")
    return {f"position.{name}": float(value)
            for name, value in zip(message.name, message.position)
            if name not in FIXED_JOINTS}


def command_positions(names, values):
    if len(names) != len(values):
        raise ValueError("Statistics names/values 长度不一致")
    result = {}
    for name, value in zip(names, values):
        if name.startswith("command_interface.") and name.endswith("/position"):
            joint = name[len("command_interface."):-len("/position")]
            if joint not in FIXED_JOINTS:
                result[f"position.{joint}"] = float(value)
    return result


class CommandDecoder:
    """Versioned names can arrive after their first values message."""

    def __init__(self):
        self.names_by_version = {}
        self.pending = defaultdict(list)

    def consume(self, topic, message, timestamp):
        if topic == NAMES_TOPIC:
            version = int(message.names_version)
            names = tuple(message.names)
            previous = self.names_by_version.get(version)
            if previous is not None and previous != names:
                raise ValueError(f"Statistics names_version={version} 对应了不同名称表")
            self.names_by_version[version] = names
            return [(source, time, command_positions(names, values))
                    for source, time, values in self.pending.pop(version, [])]
        if topic.endswith("/full"):
            statistics = message.statistics
            return [(topic, timestamp, command_positions(
                [item.name for item in statistics], [item.value for item in statistics]))]
        version = int(message.names_version)
        names = self.names_by_version.get(version)
        if names is None:
            self.pending[version].append((topic, timestamp, tuple(message.values)))
            return []
        return [(topic, timestamp, command_positions(names, message.values))]


def signal_identity(name):
    topic, _, field = name.partition(".")
    if topic in (JOINT_TOPIC, *COMMAND_TOPICS) and field.startswith("position."):
        return field[len("position."):]
    return name


def display_label(label):
    return f"{label[:2]}{signal_identity(label[2:])}"
