import docker
from enum import Enum, auto

from picrystal_test.test_platform.logger_setup import logger


class ImageStatus(Enum):
    BUILDING = auto()
    BUILT = auto()
    FAILED_TO_BUILD = auto()


class DockerClient:

    def __init__(self):
        self.client = docker.from_env()
        self.image_status = {}

    def build(self, **kwargs):
        if "tag" in kwargs:
            self.image_status[kwargs["tag"]] = ImageStatus.BUILDING
        else:
            logger.error("Image tag must be provided when building image.")
            return None

        kwargs["rm"] = True

        try:
            image, logs = self.client.images.build(**kwargs)
        except (docker.errors.BuildError, TypeError):
            self.image_status[kwargs["tag"]] = ImageStatus.FAILED_TO_BUILD
            logger.exception("Exception occurred when building image.")
            return None

        self.image_status[kwargs["tag"]] = ImageStatus.BUILT
        for line in logs:
            if 'stream' in line:
                logger.info(line['stream'].strip())
        return image

    def run(self, **kwargs):
        kwargs["detach"] = True

        try:
            container = self.client.containers.run(**kwargs)
        except (docker.errors.ContainerError, docker.errors.APIError):
            logger.exception("Exception occurred when creating container.")
            return None

        logger.info(f"Container {container.short_id} is created.")
        return container

    def logs(self, container_id: str):
        try:
            container = self.client.containers.get(container_id)
        except docker.errors.NotFound:
            return f"Container {container_id} not found."
        except docker.errors.APIError:
            logger.exception("Exception occurred when getting container logs")
            return "Got APIError when getting container logs"

        return container.logs()
