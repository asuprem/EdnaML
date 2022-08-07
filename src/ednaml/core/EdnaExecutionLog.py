import json
from typing import List


class EdnaExecutionCommand:
    command: str
    argument_key: List[str]
    argument_value: List[str]
    argument_alias: List[str]
    def __init__(self, command: str, argument_key: List[str], argument_value: List[str], argument_alias: List[str]):
        self.command = command
        self.argument_key = [item for item in argument_key]
        self.argument_value = [item for item in argument_value]
        self.argument_alias = [item for item in argument_alias]

    def __repr__(self) -> str:
        return json.dumps(
            self.__dict__, indent=2
        )

class EdnaExecutionLog:
    """The EdnaExecutionLog tracks all commands used in an EdnaML or EdnaDeploy execution.
    """
    executionLog: List[EdnaExecutionCommand]

    def __init__(self):
        self.executionLog = []

    def addCommand(self, command: str, argument_key: List[str], argument_value: List[str], argument_alias: List[str]):
        self.executionLog.append(EdnaExecutionCommand(command=command, argument_key=argument_key, argument_value=argument_value, argument_alias=argument_alias))
    
