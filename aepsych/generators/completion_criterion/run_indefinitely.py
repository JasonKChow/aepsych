#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Set

from aepsych.config import Config, ConfigurableMixin
from ax.core import Experiment
from ax.modelbridge.transition_criterion import TransitionCriterion


class RunIndefinitely(TransitionCriterion, ConfigurableMixin):
    def __init__(self, run_indefinitely: bool) -> None:
        self.run_indefinitely = run_indefinitely

    def is_met(self, experiment: Experiment) -> bool:
        return not self.run_indefinitely

    def block_continued_generation_error(
        self,
        node_name: Optional[str],
        model_name: Optional[str],
        experiment: Optional[Experiment],
        trials_from_node: Optional[Set[int]] = None,
    ) -> None:
        pass

    @classmethod
    def get_config_options(cls, config: Config, name: str) -> Dict[str, Any]:
        run_indefinitely = config.getboolean(name, "run_indefinitely", fallback=False)
        options = {"run_indefinitely": run_indefinitely}
        return options
