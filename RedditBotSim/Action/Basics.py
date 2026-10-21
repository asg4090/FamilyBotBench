
# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error

from typing import List, Optional, Callable, Dict, Any, Type
from pydantic import BaseModel

class Action(BaseModel):
    name: str
    description: str
    func: Optional[Callable[..., str]]
    input_args_schema: Optional[Dict[str, Any]] = None
    enable: bool = True
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_function(
            cls,
            func: Optional[Callable],
            name: str,  # We keep these required to support backwards compatibility
            description: str,
            input_args_schema: dict,
            # output_args_schema: dict,
            **kwargs: Any,
    ):
        return cls(
            name=name,
            func=func,
            description=description,
            input_args_schema=input_args_schema,
            # output_args_schema=output_args_schema,
            **kwargs,
        )

    def args_check(self, args, args_schema):
        for arg_name, arg_type in args_schema.items():
            if arg_name not in args:
                return f"Missing argument: {arg_name}"
            if not isinstance(args[arg_name], arg_type):
                return f"Invalid argument type for {arg_name}: {type(args[arg_name])}, expected {arg_type}"
        return None 

    def __call__(
            self,
            input_action_args: Dict[str, Any],
            env: Any,
            agent: Any,
    ) -> str:
        """Use the tool."""
        # error_message = self.args_check(input_action_args, self.input_args_schema)
        # if error_message:
        #     return error_message
        # else:
        action_output = self.func(input_action_args, env, agent)
        return action_output

class ActionSpace:
    actions: List[Action]
    action_names: List[str]

    def __repr__(self):
        return [{"name": action.name, "description": action.description, 
                 "input_args": action.input_args_schema} for action in self.actions]
    
