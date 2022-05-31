# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from distutils.util import strtobool
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.networks.nets import UNETR

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.tasks.activelearning.tta import TTA
from monailabel.tasks.scoring.dice import Dice
from monailabel.tasks.scoring.epistemic import EpistemicScoring
from monailabel.tasks.scoring.sum import Sum
from monailabel.tasks.scoring.tta import TTAScoring
from monailabel.utils.others.generic import download_file

logger = logging.getLogger(__name__)


class SegmentationCardiac(TaskConfig):
    def __init__(self):
        super().__init__()

        self.epistemic_enabled = None
        self.epistemic_samples = None
        self.tta_enabled = None
        self.tta_samples = None

    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = ["cardiac"]

        # Model Files
        model_file_name = "dc_91_best_metric_model.pth"
        self.path = [
            os.path.join(self.model_dir, model_file_name),  # pretrained
            os.path.join(self.model_dir, model_file_name),  # published
        ]

        # Download PreTrained Model
        # if strtobool(self.conf.get("use_pretrained_model", "true")):
        #     url = f"{self.NGC_PATH}/clara_pt_cardiac_ct_segmentation/versions/1/files/models/model.pt"
        #     download_file(url, self.path[0])

        # Network
        self.network = UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )

        # Others
        self.epistemic_enabled = strtobool(conf.get("epistemic_enabled", "false"))
        self.epistemic_samples = int(conf.get("epistemic_samples", "5"))
        logger.info(f"EPISTEMIC Enabled: {self.epistemic_enabled}; Samples: {self.epistemic_samples}")

        self.tta_enabled = strtobool(conf.get("tta_enabled", "false"))
        self.tta_samples = int(conf.get("tta_samples", "5"))
        logger.info(f"TTA Enabled: {self.tta_enabled}; Samples: {self.tta_samples}")

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.SegmentationCardiac(
            path=self.path, network=self.network, labels=self.labels, preload=False
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        task: TrainTask = lib.trainers.SegmentationCardiac(
            model_dir=output_dir,
            network=self.network,
            description="Train Cardiac Segmentation Model",
            load_path=self.path[0],
            publish_path=self.path[1],
            labels=self.labels,
        )
        return task

    def strategy(self) -> Union[None, Strategy, Dict[str, Strategy]]:
        strategies: Dict[str, Strategy] = {}
        if self.epistemic_enabled:
            strategies[f"{self.name}_epistemic"] = Epistemic()
        if self.tta_enabled:
            strategies[f"{self.name}_tta"] = TTA()
        return strategies

    def scoring_method(self) -> Union[None, ScoringMethod, Dict[str, ScoringMethod]]:
        methods: Dict[str, ScoringMethod] = {
            "dice": Dice(),
            "sum": Sum(),
        }

        if self.epistemic_enabled:
            methods[f"{self.name}_epistemic"] = EpistemicScoring(
                model=self.path,
                network=UNETR(
                    in_channels=1,
                    out_channels=2,
                    img_size=(96, 96, 96),
                    feature_size=16,
                    hidden_size=768,
                    mlp_dim=3072,
                    num_heads=12,
                    pos_embed="perceptron",
                    norm_name="instance",
                    res_block=True,
                    dropout_rate=0.0,
                ),
                transforms=lib.infers.SegmentationCardiac(None).pre_transforms(),
                num_samples=self.epistemic_samples,
            )
        if self.tta_enabled:
            methods[f"{self.name}_tta"] = TTAScoring(
                model=self.path,
                network=self.network,
                deepedit=False,
                num_samples=self.tta_samples,
                spatial_size=(128, 128, 64),
                spacing=(1.0, 1.0, 1.0),
            )
        return methods
