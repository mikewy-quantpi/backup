from fastapi import FastAPI, BackgroundTasks, UploadFile
from fastapi.responses import FileResponse

from picrystal_test.test_platform.docker_client import DockerClient
from picrystal_test.test_platform.use_case_util import UseCase, UseCaseStatus
from picrystal_test.test_platform.test_execution import TestExecution, TestExecutionStatus


app = FastAPI()

docker_client = DockerClient()


@app.post("/create_use_case_image/")
async def create_use_case_image(
    use_case_zip: UploadFile,
    image_tag: str,
    background_tasks: BackgroundTasks
):
    use_case = UseCase(image_tag)

    use_case.setup_image_build_environment(use_case_zip.file)
    if use_case.status == UseCaseStatus.FAILED_TO_CREATE_ENV:
        return {"message": f"Failed to build image {image_tag}. Check details in log."}

    background_tasks.add_task(use_case.build_image, docker_client)
    return {"message": f"Start to build image {image_tag}."}


@app.get("/get_image_status/{image_name}")
async def get_image_status(image_name: str):
    if image_name not in docker_client.image_status:
        return {"message": f"Image status of {image_name} not found"}
    return {"message": f"Image {image_name} is "
                       f"{docker_client.image_status[image_name].name}"}


@app.post("/execute_test_suite/")
async def execute_test_suite(
    trust_profile: UploadFile,
    image_name: str
):
    if image_name not in UseCase.registry:
        return {"message": f"Image {image_name} is not available."}

    if not trust_profile.filename.endswith(".json"):
        return {"message": f"File {trust_profile.filename} must be a JSON file."}

    test_execution = TestExecution(trust_profile.filename, UseCase.registry[image_name])
    test_execution.store_trust_profile(trust_profile.file)
    test_execution.execute_test_suite(docker_client)

    if test_execution.status == TestExecutionStatus.FAILED:
        return {"message": "Got error when executing test suite. Check detail in log"}
    return {"message": f"Test suite is running. "
                       f"Container short_id is {test_execution.container_short_id}"}


@app.get("/get_test_result_path/")
async def get_test_result_path(
    trust_profile_name: str,
    image_name: str
):
    return {
        "message": TestExecution.get_test_result_path(trust_profile_name, image_name)
    }


@app.get("/download_test_result/")
async def download_test_result(
    trust_profile_name: str,
    image_name: str
):
    if not TestExecution.is_test_result_ready(trust_profile_name, image_name):
        return {"message": "Test result is not available."}

    return FileResponse(
        path=TestExecution.get_test_result_path(trust_profile_name, image_name),
        media_type="application/json"
    )


@app.get("/check_container_logs/")
async def check_container_logs(container_id: str):
    return {"message": docker_client.logs(container_id)}
