##
##
##

import json
from pathlib import Path
from dataset import H2ODataset
from PIL import Image
from tqdm import tqdm


def generate_detections(
    dataset_path: Path,
    split: str,
    detection_path: Path,
) -> None:
    """Generates detections for the given dataset."""
    categories = H2ODataset._get_entity_classes(dataset_path)
    entity_category_to_id = {name: i for i, name in enumerate(categories)}
    samples = H2ODataset._get_samples(dataset_path, split)

    for sample in tqdm(samples, desc=f"Generating detections for {split} split"):
        id = sample["id"]
        image_path = dataset_path / "images" / split / f"{id}.jpg"
        image = Image.open(image_path)
        H, W = image.height, image.width

        boxes = []
        labels = []
        scores = []

        for entity in sample["entities"]:
            box = entity["bbox"]
            # convert from normalized coordinates to absolute coordinates
            box = [box[0] * W, box[1] * H, box[2] * W, box[3] * H]
            boxes.append(box)

            labels.append(entity_category_to_id[entity["category"]])
            scores.append(1.0)

        detections = {"boxes": boxes, "labels": labels, "scores": scores}
        file_path = detection_path / split / f"{id}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w+") as file:
            json.dump(detections, file)


def main() -> None:
    dataset_path = Path("datasets/h2o")
    detection_path = Path("detections/h2o")

    generate_detections(dataset_path, "train", detection_path)
    generate_detections(dataset_path, "test", detection_path)


if __name__ == "__main__":
    main()
