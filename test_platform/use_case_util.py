"""
This file is named use_case_util.py, mainly to distinguish it from use_case.py
that is used to provide custom use case.
"""
import os
import uuid
import zipfile
from enum import Enum, auto
from typing import BinaryIO

import shortuuid
from picrystal_test.test_platform.config import (CONTAINER_ROOT_PATH,
                                                 DOCKERFILE_TEMPLATE,
                                                 ROOT_PATH,
                                                 USE_CASE_RELATIVE_PATH_TAG)
from picrystal_test.test_platform.docker_client import ImageStatus
from picrystal_test.test_platform.logger_setup import logger


class UseCaseStatus(Enum):
    INITIALIZED = auto()
    FAILED_TO_CREATE_ENV = auto()
    ENV_CREATED = auto()
    START_TO_BUILD_IMAGE = auto()
    FAILED_TO_BUILD_IMAGE = auto()
    IMAGE_BUILT = auto()


class UseCase:
    """
    Setting up image build environment and building use case docker image

    Here is one example file tree on the host ROOT_PATH:

    ~/test-framework
    ├── custom_usecases
    │   ├── 123456
    │   │   ├── Dockerfile
    │   │   ├── requirements.txt
    │   │   ├── results
    │   │   │   └── test_result.json
    │   │   └── use_case.py
    │   ├── abcdef
    │   │   ├── Dockerfile
    │   │   ├── requirements.txt
    │   │   ├── results
    │   │   │   └── test_result.json
    │   │   └── use_case.py
    │   └── other_usecases
    ├── dockerfile
    │   ├── Dockerfile
    │   └── template
    ├── picrystal_test
    │   └── core.py
    ├── setup.py
    └── usecases
        ├── credit_default.py
        └── results

    And on container, there is only its own custom_usecases directory:
    ~/app
    ├── custom_usecases
    │   ├── 123456
    │   │   ├── Dockerfile
    │   │   ├── requirements.py
    │   │   ├── results
    │   │   │   └── test_result.json
    │   │   └── use_case.py
    ├── picrystal_test
    │   └── core.py
    ├── setup.py
    """

    CUSTOM_USECASES = "custom_usecases"
    RESULTS = "results"

    registry = {}

    def __init__(self, image_tag):
        self.root_path = ROOT_PATH
        self.use_case_relative_path = None
        self.use_case_path = None
        self.dockerfile_relative_path = None  # This is a file instead of a directory
        self.container_use_case_path = None
        self.image_tag = image_tag
        self.status = UseCaseStatus.INITIALIZED
        self.registry[image_tag] = self

    def setup_image_build_environment(self, use_case_zip: BinaryIO):
        try:
            self._make_directories()
            self._unzip_files(use_case_zip)
            self._generate_dockerfile()
        except OSError:
            logger.exception("Exception occurred when setting up image build environment")
            self.status = UseCaseStatus.FAILED_TO_CREATE_ENV
        self.status = UseCaseStatus.ENV_CREATED

    def build_image(self, docker_client):
        image_config = {
            "tag": self.image_tag,
            "path": self.root_path,
            "dockerfile": self.dockerfile_relative_path
        }
        docker_client.build(**image_config)
        if docker_client.image_status[self.image_tag] == ImageStatus.BUILT:
            self.status = UseCaseStatus.IMAGE_BUILT
        else:
            self.status = UseCaseStatus.FAILED_TO_BUILD_IMAGE

    def _make_directories(self):
        """
        self.root_path = ~/test-framework
        self.use_case_relative_path = custom_usecases/123456
        self.use_case_path = ~/test-framework/custom_usecases/123456
        self.container_use_case_path = /app/custom_usecases/123456
        """
        unique_id = uuid.uuid4()
        short_id = shortuuid.encode(unique_id)[:12]

        self.use_case_relative_path = self.CUSTOM_USECASES + "/" + short_id
        self.use_case_path = os.path.join(self.root_path, self.use_case_relative_path)
        results_path = os.path.join(self.use_case_path, self.RESULTS)

        os.makedirs(self.use_case_path)
        os.makedirs(results_path)

        self.container_use_case_path = os.path.join(
            CONTAINER_ROOT_PATH,
            self.use_case_relative_path,
        )

    def _unzip_files(self, zip_data: BinaryIO):
        """
        self.use_case_path = ~/test-framework/custom_usecases/123456
        zip file content:
            - requirements.txt
            - use_case.py
        """
        with zipfile.ZipFile(zip_data, "r") as zip_ref:
            zip_ref.extractall(self.use_case_path)

    def _generate_dockerfile(self):
        """
        - Read Dockerfile template from ~/test-framework/dockerfile/template
        - self.use_case_relative_path = custom_usecases/123456
        - self.dockerfile_relative_path = custom_usecases/123456/Dockerfile
        - Write Dockerfile to ~/test-framework/custom_usecases/123456/Dockerfile
        """
        template_path = os.path.join(self.root_path, DOCKERFILE_TEMPLATE)
        with open(template_path, "r") as file:
            template_content = file.read()

        dockerfile_content = template_content.replace(
            USE_CASE_RELATIVE_PATH_TAG,
            self.use_case_relative_path
        )

        self.dockerfile_relative_path = os.path.join(
            self.use_case_relative_path,
            "Dockerfile"
        )

        dockerfile_path = os.path.join(self.root_path, self.dockerfile_relative_path)
        with open(dockerfile_path, "w") as file:
            file.write(dockerfile_content)
