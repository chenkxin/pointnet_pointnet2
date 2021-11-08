import torch

from smnist import SMNIST_Dataset
from classification_dataset import ClassificationDataset
from transforms import CacheNPY, ProjectOnSphere, ToMesh, get_transform


def _get_classes(dataset_name):
    if dataset_name == "modelnet10":
        classes = modelnet10_classes
    elif dataset_name == "modelnet40":
        classes = modelnet40_classes
    elif dataset_name == "shrec15_0.2":
        classes = shrec15_classes
    elif dataset_name == "shrec15_0.3":
        classes = shrec15_classes
    elif dataset_name == "shrec15_0.4":
        classes = shrec15_classes
    elif dataset_name == "shrec15_0.5":
        classes = shrec15_classes
    elif dataset_name == "shrec15_0.6":
        classes = shrec15_classes
    elif dataset_name == "shrec15_0.7":
        classes = shrec15_classes
    elif dataset_name == "shrec17":
        classes = shrec17_classes
    elif dataset_name == "smnist":
        classes = [str(i) for i in range(10)]
    else:
        raise ValueError(f"No such dataset {dataset_name}")
    return classes


def makeDataset(
    train=True,
    b=32,
    dataset_name="modelnet10",
    root="~/caps3d/data",
    **kwargs,
):
    # spherical images
    if dataset_name == "smnist":
        return SMNIST_Dataset(
            no_rotate_train=kwargs["no_rotate_train"],
            no_rotate_test=kwargs["no_rotate_test"],
            train=train,
            overlap=kwargs["overlap"],
        )

    # 3d dataset
    classes = _get_classes(dataset_name)

    def target_transform(x):
        return classes.index(x)

    return ClassificationDataset(
        root=root,
        dataset_name=dataset_name,
        train=train,
        b=b,
        target_transform=target_transform,
        **kwargs,
    )


def makeDataLoader(
    train=True,
    bw=32,
    batch_size=4,
    num_workers=4,
    dataset_name="modelnet10",
    root="/home/chenkangxin/Pointnet_Pointnet2_pytorch/data",
    **kwargs,
):

    classes = _get_classes(dataset_name)
    dataset = makeDataset(train, bw, dataset_name, root, **kwargs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return len(dataset), classes, dataloader


modelnet10_classes = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]
modelnet40_classes = [
    "tv_stand",
    "guitar",
    "lamp",
    "cup",
    "bed",
    "desk",
    "vase",
    "bottle",
    "bookshelf",
    "chair",
    "tent",
    "sink",
    "curtain",
    "wardrobe",
    "glass_box",
    "door",
    "range_hood",
    "mantel",
    "dresser",
    "plant",
    "stairs",
    "bench",
    "bowl",
    "night_stand",
    "table",
    "flower_pot",
    "airplane",
    "cone",
    "xbox",
    "radio",
    "laptop",
    "bathtub",
    "monitor",
    "person",
    "toilet",
    "car",
    "stool",
    "keyboard",
    "piano",
    "sofa",
]
shrec15_classes = [
    "sumotori",
    "paper",
    "bull",
    "mouse",
    "horse",
    "man",
    "aligator",
    "nunchaku",
    "robot",
    "santa",
    "dinosaur",
    "hand",
    "armadillo",
    "spider",
    "frog",
    "alien",
    "tortoise",
    "twoballs",
    "ants",
    "dragon",
    "mantaray",
    "elephant",
    "lamp",
    "ring",
    "watch",
    "woman",
    "octopus",
    "weedle",
    "centaur",
    "dinoske",
    "snake",
    "woodman",
    "deer",
    "glasses",
    "mermaid",
    "pliers",
    "kangaroo",
    "bird",
    "dog",
    "giraffe",
    "rabbit",
    "chick",
    "camel",
    "shark",
    "gorilla",
    "cat",
    "flamingo",
    "scissor",
]
shrec17_classes = [
    "Pillow",
    "Box",
    "Machine",
    "Cup",
    "Sofa",
    "Bin",
    "Storage",
    "Bed",
    "Chair",
    "Light",
    "Desk",
    "PCcase",
    "Table",
    "Bookshelf",
    "Oven",
    "Bag",
    "Printer",
    "Display",
    "Book",
    "Keyboard",
]

if __name__ == '__main__':
    makeDataLoader()