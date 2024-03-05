# Test Framework Platform

## Usage and Architecture
- [This Miro Board](https://miro.com/app/board/uXjVMrkef2E=/?share_link_id=751045067182) briefly describes the usage and architecture of **Test Framework Platform**. In general, it serves for two purposes:
  - Build docker image for individual machine learning use case together with the test framework
  - Upload a trust profile, specify a use case image, and execute test suite

## How to host the Test Framework Platform service
```
cd /tmp
git clone https://github.com/QuantPi/test-framework.git
cd test-framework/picrystal_test/test_platform/deployment/gcp
./deploy.sh -h
./deploy.sh
```

## Interact with the Test Framework Platform through API endpoints
- The test framework platform exposes some API endpoints for the functions mentioned above.
- Here is [the link](http://34.32.196.244/redoc) to the API endpoints document.

### Create use case image
```
curl -X POST \
   -F "use_case_zip=@/tmp/picrystal/create_use_case_image_test/use_case.zip" \
   http://<hostname>/create_use_case_image/\?image_tag\=use_case_test:0.0.7
```
- use_case.zip contains two files:
  - `use_case.py` provides machine learning model, test dataset, and specific test operation functions
  - `requirements.txt` lists the pip requirements of the use case
  - These two files in the zip file must be named in this way. Otherwise it would cause issues when generating the use case Dockerfile. The zip file itself can be named freely.
- check out one example use_case.zip in `example` folder 
- image_tag can be any tag that fulfills docker image naming convention.

### Get image status
```
watch -n 1 curl http://<hostname>/get_image_status/use_case_test:0.0.7

# If you have access to the server
watch -n 1 docker images
```

### Execute test suite
```
curl -X POST \
   -F "trust_profile=@/tmp/picrystal/execute_test_suite_test/trust_profile.json" \
   http://<hostname>/execute_test_suite/\?image_name\=use_case_test:0.0.7
```
- Trust profile must be a json file. Apart from that, there is no restriction on its file name. In fact, different trust profile represents different test suite, while they could be used to test against the same use case. Test results of different trust profiles are stored as different json files in the same use case directory, if these test suites are executed against the same machine learning use case.
- check out one example trust_profile.json in `example` folder

### Download test result
```
curl http://<hostname>/download_test_result/\?trust_profile_name\=trust_profile.json\&image_name\=use_case_test:0.0.7
```
- This API will return the path of the result json file on the host, which can be consumed by visualization component, analytics solutions, or any downstream applications.

## Interact with the Test Framework Platform through CLI
- The CLI is written in Go, and the source code in the `cli` folder
- It would be better to build the CLI executable by yourself to ensure its consistency with your laptop hardware architecture and operating system
- The CLI has self-explanable help document, and its usage is very similar with the API endpoints
- One pre-requisite of using it the CLI is to set this environment variable to the IP address of the test framework platform server.
```
export $PICRYSTAL_HOST_URL=192.168.0.1
```

## File organization for use cases
```
custom_usecases
├── 9y3LJkGpFZ5a            # after image build
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── results
│   └── use_case.py
├── WCvmLyVo3hDQ            # also after image build
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── results
│   └── use_case.py
└── Ytuih2cKSBt6            # after test suite execution
    ├── Dockerfile
    ├── requirements.txt
    ├── results
    │   └── test_result_trust_profile.json
    ├── trust_profile.json
    └── use_case.py
```
- Each use case is stored in seperate folders named with a 12 characters long short uuid. During test execution, docker container binds volumes only with its corresponding use case directory on the host, namely the folder named with the uuid.

## Test framework platform logs
- All activities occurred with test framework platform are logged in file `test_platform.log`, which is mainly used for dignosing issues.
- Container logs are not included. User could use `docker logs  <container_short_id>` to check each container logs. User could also use api endpoint `check_container_logs` or CLI command `qpi logs`to check the log.
