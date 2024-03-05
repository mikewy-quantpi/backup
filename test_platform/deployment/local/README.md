# Deploy Test Framework Platform locally

## Motivation
- avoid attack to cloud server from Internet
- easy to play with the test framework

## Why not do it automatically?
- The docker-in-docker approach works fine with the cloud deployment. However, we face some problems in both Mac and Windows. In addition, it is not very easy to unify the deployment on different operating systems and hardware architecture. Therefore, we choose to document the steps to deploy it manually.

## How to deploy it locally?
- This manual procedure works for Mac. For Windows, it might be slightly different.
```
# Change to the directory where you want to deploy test-framework
cd /tmp

# git clone the test-framework repository
git clone git@github.com:QuantPi/test-framework.git

# Set environment variable PICRYSTAL_ROOT_PATH
export PICRYSTAL_ROOT_PATH=/tmp/test-framework

# Create a python virtual environment and activate it
python -m venv ~/venvs/platform
source ~/venvs/platform/bin/activate

# Install the picrystal_test package
cd test-framework
pip install -e .

# Install the test_platform requirements
cd picrystal_test/test_platform
pip install -r requirements.txt

# Start the api_server
uvicorn api_server:app --host 0.0.0.0 --port 80

# The rest steps need to be carried out in another terminal session,
# since we have run the api server as a foreground job in the old session.

# If you want to use the qpi CLI, set environment variable PICRYSTAL_HOST_URL
export PICRYSTAL_HOST_URL=127.0.0.1

# Change directory to the example directory to build use case image and execute test
cd /tmp/test-framework/picrystal_test/test_platform/examples/hiring
```
