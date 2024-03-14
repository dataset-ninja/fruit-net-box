from typing import Dict, List, Literal, Optional, Union

from dataset_tools.templates import (
    AnnotationType,
    Category,
    CVTask,
    Domain,
    Industry,
    License,
    Research,
)

##################################
# * Before uploading to instance #
##################################
PROJECT_NAME: str = "FruitNet and FruitBox Dataset"
PROJECT_NAME_FULL: str = "FruitNet and FruitBox Dataset"
HIDE_DATASET = True  # set False when 100% sure about repo quality

##################################
# * After uploading to instance ##
##################################
LICENSE: License = License.Unknown()
APPLICATIONS: List[Union[Industry, Domain, Research]] = [Industry.Food(), Industry.Robotics()]
CATEGORY: Category = Category.Food(extra=Category.Robotics())

CV_TASKS: List[CVTask] = [CVTask.ObjectDetection(), CVTask.Identification()]
ANNOTATION_TYPES: List[AnnotationType] = [AnnotationType.ObjectDetection()]

RELEASE_DATE: Optional[str] = None  # e.g. "YYYY-MM-DD"
if RELEASE_DATE is None:
    RELEASE_YEAR: int = 2023

HOMEPAGE_URL: str = "https://www.kaggle.com/datasets/mirlab/annotated-fruitnet-and-fruitbox"
# e.g. "https://some.com/dataset/homepage"

PREVIEW_IMAGE_ID: int = 15663812
# This should be filled AFTER uploading images to instance, just ID of any image.

GITHUB_URL: str = "https://github.com/dataset-ninja/fruit-net-box"
# URL to GitHub repo on dataset ninja (e.g. "https://github.com/dataset-ninja/some-dataset")

##################################
### * Optional after uploading ###
##################################
DOWNLOAD_ORIGINAL_URL: Optional[Union[str, dict]] = (
    "https://www.kaggle.com/datasets/mirlab/annotated-fruitnet-and-fruitbox/download?datasetVersionNumber=2"
)
# Optional link for downloading original dataset (e.g. "https://some.com/dataset/download")

CLASS2COLOR: Optional[Dict[str, List[str]] or Literal["predefined"]] = {
    "good apple": [230, 25, 75],
    "good orange": [60, 180, 75],
    "bad apple": [255, 225, 25],
    "bad orange": [0, 130, 200],
    "bad guava": [245, 130, 48],
    "good guava": [145, 30, 180],
    "fruit box": [70, 240, 240],
}
# If specific colors for classes are needed, fill this dict (e.g. {"class1": [255, 0, 0], "class2": [0, 255, 0]})

# If you have more than the one paper, put the most relatable link as the first element of the list
# Use dict key to specify name for a button
PAPER: Optional[Union[str, List[str], Dict[str, str]]] = (
    "https://www.researchgate.net/publication/374803244_Multimodal_Siamese_Framework_for_Accurate_Grade_and_Measure_Estimation_of_Tropical_Fruits"
)
BLOGPOST: Optional[Union[str, List[str], Dict[str, str]]] = None
REPOSITORY: Optional[Union[str, List[str], Dict[str, str]]] = None

CITATION_URL: Optional[str] = None
AUTHORS: Optional[List[str]] = ["Misaj Sharafudeen", "Vinod Chandra"]
AUTHORS_CONTACTS: Optional[List[str]] = ["vchandra@ee.iitd.ac.in", "misaj.sharafudeen@gmail.com"]

ORGANIZATION_NAME: Optional[Union[str, List[str]]] = "University of Kerala, India"
ORGANIZATION_URL: Optional[Union[str, List[str]]] = "https://www.keralauniversity.ac.in/home"

# Set '__PRETEXT__' or '__POSTTEXT__' as a key with string value to add custom text. e.g. SLYTAGSPLIT = {'__POSTTEXT__':'some text}
SLYTAGSPLIT: Optional[Dict[str, Union[List[str], str]]] = {
    "__POSTTEXT__": "Additionally, some images marked with its ***weight*** tag"
}
TAGS: Optional[
    List[
        Literal[
            "multi-view",
            "synthetic",
            "simulation",
            "multi-camera",
            "multi-modal",
            "multi-object-tracking",
            "keypoints",
            "egocentric",
        ]
    ]
] = None


SECTION_EXPLORE_CUSTOM_DATASETS: Optional[List[str]] = None

##################################
###### ? Checks. Do not edit #####
##################################


def check_names():
    fields_before_upload = [PROJECT_NAME]  # PROJECT_NAME_FULL
    if any([field is None for field in fields_before_upload]):
        raise ValueError("Please fill all fields in settings.py before uploading to instance.")


def get_settings():
    if RELEASE_DATE is not None:
        global RELEASE_YEAR
        RELEASE_YEAR = int(RELEASE_DATE.split("-")[0])

    settings = {
        "project_name": PROJECT_NAME,
        "project_name_full": PROJECT_NAME_FULL or PROJECT_NAME,
        "hide_dataset": HIDE_DATASET,
        "license": LICENSE,
        "applications": APPLICATIONS,
        "category": CATEGORY,
        "cv_tasks": CV_TASKS,
        "annotation_types": ANNOTATION_TYPES,
        "release_year": RELEASE_YEAR,
        "homepage_url": HOMEPAGE_URL,
        "preview_image_id": PREVIEW_IMAGE_ID,
        "github_url": GITHUB_URL,
    }

    if any([field is None for field in settings.values()]):
        raise ValueError("Please fill all fields in settings.py after uploading to instance.")

    settings["release_date"] = RELEASE_DATE
    settings["download_original_url"] = DOWNLOAD_ORIGINAL_URL
    settings["class2color"] = CLASS2COLOR
    settings["paper"] = PAPER
    settings["blog"] = BLOGPOST
    settings["repository"] = REPOSITORY
    settings["citation_url"] = CITATION_URL
    settings["authors"] = AUTHORS
    settings["authors_contacts"] = AUTHORS_CONTACTS
    settings["organization_name"] = ORGANIZATION_NAME
    settings["organization_url"] = ORGANIZATION_URL
    settings["slytagsplit"] = SLYTAGSPLIT
    settings["tags"] = TAGS

    settings["explore_datasets"] = SECTION_EXPLORE_CUSTOM_DATASETS

    return settings
