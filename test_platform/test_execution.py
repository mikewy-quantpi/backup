import os
from enum import Enum, auto
from typing import BinaryIO

from picrystal_test.test_platform.use_case_util import UseCase


class TestExecutionStatus(Enum):
    INITIALIZED = auto()
    EXECUTING = auto()
    FAILED = auto()
    FINISHED = auto()


class TestExecution:

    RESULTS_PATH_PREFIX = "results/test_result_"

    registry = {}

    def __init__(self, trust_profile_name: str, use_case: UseCase):
        self.trust_profile_name = trust_profile_name
        self.use_case = use_case
        self.container_short_id = None
        self.status = TestExecutionStatus.INITIALIZED
        self.output_relative_path = self.RESULTS_PATH_PREFIX + self.trust_profile_name
        self.test_result_path = os.path.join(
            self.use_case.use_case_path,
            self.output_relative_path
        )
        self.registry[use_case.image_tag] = {self.trust_profile_name: self}

    def store_trust_profile(self, trust_profile_data: BinaryIO):
        trust_profile_path = os.path.join(
            self.use_case.use_case_path,
            self.trust_profile_name
        )
        with open(trust_profile_path, "wb") as file:
            file.write(trust_profile_data.read())

    def execute_test_suite(self, docker_client):
        container_config = {
            "image": self.use_case.image_tag,
            "volumes": {
                self.use_case.use_case_path: {
                    "bind": self.use_case.container_use_case_path
                }
            },
            "command": ["--trust-profile", self.trust_profile_name,
                        "--output", self.output_relative_path]
        }
        container = docker_client.run(**container_config)

        if container is None:
            self.status = TestExecutionStatus.FAILED
        else:
            self.status = TestExecutionStatus.EXECUTING
            self.container_short_id = container.short_id

        return container

    @classmethod
    def get_test_result_path(cls, trust_profile_name: str, image_name: str):
        try:
            return cls.registry[image_name][trust_profile_name].test_result_path
        except KeyError:
            return f"Either {image_name} or {trust_profile_name} does not exist"

    @classmethod
    def is_test_result_ready(cls, trust_profile_name: str, image_name: str):
        try:
            if os.path.exists(
                cls.registry[image_name][trust_profile_name].test_result_path
            ):
                return True
        except KeyError:
            pass

        return False
