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
from typing import Callable, Sequence

from monai.inferers import Inferer, SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    Flipd,
    SqueezeDimd,
    ToNumpyd,
    ToTensord,
)

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import BoundingBoxd, Restored


class SegmentationCardiac(InferTask):
    """
    This provides Inference Engine for pre-trained cardiac segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
            self,
            path,
            network=None,
            type=InferType.SEGMENTATION,
            labels="cardiac",
            dimension=3,
            description="A pre-trained model for volumetric (3D) segmentation of the cardiac from CT image",
            **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear"),
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            EnsureTyped(keys="image"),
            ToTensord(keys=["image"]),
        ]

    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.8)

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            AsDiscreted(keys="pred", argmax=True),
            Orientationd(keys=["pred"], axcodes="LPS"),
            ToNumpyd(keys="pred"),
            Restored(keys=["pred"], ref_image="image"),
            BoundingBoxd(keys="pred", result="result", bbox="bbox")
        ]
