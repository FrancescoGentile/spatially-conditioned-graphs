##
##
##

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypedDict

from PIL import Image
import torch
import torchvision
from typing_extensions import NotRequired


class H2ODataset:
    """The Human-to-Human-or-Object (H2O) Interaction Dataset."""

    def __init__(self, path: Path | str, split: Literal["train", "test"]) -> None:
        """Initializes a new H2O dataset.

        Args:
            path: The path to the dataset. This should be a directory containing
                the following files:
                - `images/`: A directory containing the images. This directory
                    should be further split into a `train/` and `test/` directory.
                - `{split}.json`: A JSON file for each split containing the
                    annotations for the samples in the split.
                - `categories.json`: A JSON file containing the names of the entity
                    classes.
                - `verbs.json`: A JSON file containing the names of the interaction
                    classes.
            split: The split of the dataset to load. At the moment, only the `train` and
                `test` splits are supported.
        """
        self._path = Path(path)
        self._split = split

        self._samples = self._get_samples(self._path, self._split)
        entity_classes = self._get_entity_classes(self._path)
        self._entity_class_to_id = {name: i for i, name in enumerate(entity_classes)}

        interaction_classes = self._get_interaction_classes(self._path)
        self._interaction_class_to_id = {
            name: i for i, name in enumerate(interaction_classes)
        }

    # ---------------------------------------------------------------------- #
    # Properties
    # ---------------------------------------------------------------------- #

    @property
    def num_entity_classes(self) -> int:
        """The number of entity classes."""
        return len(self._entity_class_to_id)

    @property
    def num_interaction_classes(self) -> int:
        """The number of interaction classes."""
        return len(self._interaction_class_to_id)

    @property
    def human_class_id(self) -> int:
        """The class ID for human entities."""
        return self._entity_class_to_id["person"]

    # ---------------------------------------------------------------------- #
    # Public Methods
    # ---------------------------------------------------------------------- #

    def filename(self, index: int) -> str:
        """Returns the filename of the image at the given index."""
        return self._samples[index]["id"] + ".jpg"

    def get_object_valid_interactions(
        self,
        splits: Iterable[Literal["train", "test"]],
    ) -> list[list[int]]:
        """Return for each entity class the interactions in which it can participate.

        !!! warning

            Differently from HICO-DET, the H2O dataset does not provide the list of
            interaction classes that are valid for a given object, thus this method
            computes this list from the dataset annotations.

        Args:
            splits: The splits to consider when computing the valid interactions.

        Returns:
            A list of lists. The list at index `i` contains the interaction
            classes in which the entity class with ID `i` can participate.
        """
        valid_interactions = [set() for _ in range(self.num_entity_classes)]

        for split in splits:
            samples = self._get_samples(self._path, split)
            for sample in samples:
                for action in sample["actions"]:
                    object_ = action.get("target", action.get("instrument", None))

                    if object_ is None:
                        continue

                    object_class = sample["entities"][object_]["category"]
                    object_class_id = self._entity_class_to_id[object_class]
                    interaction_class_id = self._interaction_class_to_id[action["verb"]]

                    valid_interactions[object_class_id].add(interaction_class_id)

        return [list(interactions) for interactions in valid_interactions]

    # ---------------------------------------------------------------------- #
    # Magic Methods
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        file_sample = self._samples[index]

        image_path = self._path / f"images/{self._split}/{file_sample['id']}.jpg"
        image = Image.open(image_path).convert("RGB")

        coords = [e["bbox"] for e in file_sample["entities"]]
        entity_boxes = torch.as_tensor(coords, dtype=torch.float)
        # coordinates are normalized to [0, 1], so we need to scale them
        h, w = image.height, image.width
        entity_boxes *= torch.as_tensor([w, h, w, h], dtype=torch.float)[None]

        entity_labels = torch.as_tensor(
            [self._entity_class_to_id[e["category"]] for e in file_sample["entities"]]
        )

        actions: list[tuple[int, int]] = []
        action_labels: list[int] = []
        for action in file_sample["actions"]:
            subject = action["subject"]
            object_ = action.get("target", action.get("instrument", None))

            # we only care about interactions
            if object_ is None:
                continue

            actions.append((subject, object_))
            action_labels.append(self._interaction_class_to_id[action["verb"]])

        if len(actions) == 0:
            actions = torch.empty((0, 2), dtype=torch.long)
            action_labels = torch.empty((0,), dtype=torch.long)
        else:
            actions = torch.as_tensor(actions, dtype=torch.long)  # (A, 2)
            action_labels = torch.as_tensor(action_labels, dtype=torch.long)  # (A,)

        subject_boxes = entity_boxes[actions[:, 0]]  # (A, 4)
        object_boxes = entity_boxes[actions[:, 1]]  # (A, 4)
        object_labels = entity_labels[actions[:, 1]]  # (A,)

        target = {
            "boxes_h": subject_boxes,
            "boxes_o": object_boxes,
            "object": object_labels,
            "labels": action_labels,
        }

        return image, target

    def __str__(self) -> str:
        return "H2O Dataset"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(split={self._split}, num_samples={len(self)})"
        )

    # ---------------------------------------------------------------------- #
    # Private Methods
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _get_samples(path: Path, split: str) -> list[FileSample]:
        with open(path / f"{split}.json") as file:
            data = json.load(file)

        return [s for s in data if len(s["entities"]) > 0]

    @staticmethod
    def _get_entity_classes(path: Path) -> list[str]:
        with open(path / "categories.json") as file:
            return json.load(file)

    @staticmethod
    def _get_interaction_classes(path: Path) -> list[str]:
        with open(path / "verbs.json") as file:
            return json.load(file)


# -------------------------------------------------------------------------- #
# Data classes
# -------------------------------------------------------------------------- #


class Entity(TypedDict):
    bbox: list[float]
    category: str


class Actions(TypedDict):
    subject: int
    target: NotRequired[int]
    instrument: NotRequired[int]
    verb: str


class FileSample(TypedDict):
    id: str
    entities: list[Entity]
    actions: list[Actions]
