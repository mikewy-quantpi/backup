import os


ROOT_PATH = os.environ.get("PICRYSTAL_ROOT_PATH")
CONTAINER_ROOT_PATH = "/app"

# place holder used in Docker_template file
USE_CASE_RELATIVE_PATH_TAG = "<custom_usecases>"

# template used to generate use case Dockerfile
DOCKERFILE_TEMPLATE = "picrystal_test/test_platform/dockerfile/Dockerfile_template"
