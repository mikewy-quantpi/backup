#!/bin/bash

# Help documentation for the script
show_help() {
    echo "Usage: ./deploy.sh [OPTIONS]"
    echo "Description: Deploy test framework platform service on a GCP virtual machine"
    echo ""
    echo "Pre-requisite:"
    echo "- gcloud is installed and authenticated"
    echo ""
    echo "Options:"
    echo "-h, --help        Show this help message and exit"
    echo "-a, --address     Use an existing static IP address to create virtual machine"
    echo "There is no other options"
    echo ""
}

if !command -v gcloud &> /dev/null; then
    echo "Error: gcloud is not installed. Please install and authenticate."
    exit 1
fi

EXTERNAL_IP=""

# Parse the command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -a|--address)
            EXTERNAL_IP="$2"
            shift
            ;;
        *)
            echo "Unknown option: $key"
            show_help
            exit 1
            ;;
    esac
    shift
done

REGION=europe-west4
ZONE=europe-west4-a

INSTANCE_IP_NAME=credit-default-test-framework-platform-ip

INSTANCE_NAME=credit-default-test-framework-platform
MACHINE_TYPE=n2-standard-2  # c3-standard-4
IMAGE_PROJECT=cos-cloud
IMAGE_FAMILY=cos-stable
BOOT_DISK_SIZE=200GB
# BOOT_DISK_TYPE=

LOG_FILE=deploy_log.txt

# Define a function to add timestamp to each line of the log
add_to_log() {
    while IFS= read -r line; do
        timestamp=$(date +"%Y-%m-%d %H:%M:%S")
        echo "[$timestamp] $line" | tee -a $LOG_FILE
    done
}

# Quit if the test framework platform instance already exists on GCP
if gcloud compute instances list --format="table(NAME)" | grep $INSTANCE_NAME; then
    echo "The GCP instance ${INSTANCE_NAME} already exists. Quit deployment." | add_to_log
    exit 1
fi

# Reserve a static IP if user does not specify one
if [ "$EXTERNAL_IP" = "" ]; then
    gcloud compute addresses create $INSTANCE_IP_NAME \
        --region=$REGION \
        >> "$LOG_FILE" 2>&1

    EXTERNAL_IP=$(gcloud compute addresses describe $INSTANCE_IP_NAME \
                    --format "value(address)" \
                    --region $REGION)

    echo "Static IP ${INSTANCE_IP_NAME} with address ${EXTERNAL_IP} is created." | add_to_log
fi


# Create a virtual machine in GCP
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --address=$EXTERNAL_IP \
    --boot-disk-size=$BOOT_DISK_SIZE \
#    --boot-disk-type=$BOOT_DISK_TYPE \
    >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "last step failed. quit deployment." | add_to_log
    exit 1
fi

# Wait for vm to warm up, otherwise scp and ssh would not work
sleep 10

echo "virtual machine ${INSTANCE_NAME} is created" | add_to_log

# Copy code repository to the virtual machine
gcloud compute \
    scp --recurse /tmp/test-framework $INSTANCE_NAME:/tmp \
    --zone=$ZONE \
    >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "last step failed. quit deployment." | add_to_log
    exit 1
fi

echo "Test framework code repository is copied to the virtual machine." | add_to_log

# Start platform service
gcloud compute ssh $INSTANCE_NAME \
    --zone=$ZONE \
    --command="cd /tmp/test-framework; git checkout main; docker build -t platform:0.0.1 -f picrystal_test/test_platform/deployment/gcp/Dockerfile ." \
    >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "last step failed. quit deployment." | add_to_log
    exit 1
fi

echo "Docker image of test framework platform is built with tag platform:0.0.1." | add_to_log

gcloud compute ssh $INSTANCE_NAME \
    --zone=$ZONE \
    --command="docker run -d -v /var/run/docker.sock:/var/run/docker.sock -v /tmp/test-framework/custom_usecases:/tmp/test-framework/custom_usecases -p 80:80 platform:0.0.1" \
    >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "last step failed. quit deployment." | add_to_log
    exit 1
fi

echo "Test framework platform service is up and running." | add_to_log

# Start dashboard service
gcloud compute ssh $INSTANCE_NAME \
    --zone=$ZONE \
    --command="cd /tmp/test-framework; git checkout main; docker build -t dashboard:0.0.1 -f picrystal_test/test_platform/deployment/gcp/Dockerfile_dashboard ." \
    >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "last step failed. quit deployment." | add_to_log
    exit 1
fi

echo "Docker image of dashboard is built with tag dashboard:0.0.1." | add_to_log

gcloud compute ssh $INSTANCE_NAME \
    --zone=$ZONE \
    --command="docker run -d -v /var/run/docker.sock:/var/run/docker.sock -v /tmp/test-framework:/tmp/test-framework -p 8080:8080 dashboard:0.0.1" \
    >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "last step failed. quit deployment." | add_to_log
    exit 1
fi

echo "Dashboard service is up and running." | add_to_log

# Start credit dashboard service
gcloud compute ssh $INSTANCE_NAME \
    --zone=$ZONE \
    --command="cd /tmp/test-framework; git checkout main; docker build -t credit_dashboard:0.0.1 -f picrystal_test/test_platform/deployment/gcp/Dockerfile_credit_dashboard ." \
    >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "last step failed. quit deployment." | add_to_log
    exit 1
fi

echo "Docker image of credit default dashboard is built with tag credit_dashboard:0.0.1." | add_to_log

gcloud compute ssh $INSTANCE_NAME \
    --zone=$ZONE \
    --command="docker run -d -v /var/run/docker.sock:/var/run/docker.sock -v /tmp/test-framework:/tmp/test-framework -p 9090:9090 credit_dashboard:0.0.1" \
    >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "last step failed. quit deployment." | add_to_log
    exit 1
fi

echo "Dashboard service is up and running." | add_to_log
# Output the docs endpoint of the service
DOCS_URL="http://${EXTERNAL_IP}/redoc/"
DASHBOARD_URL="http://${EXTERNAL_IP}:8080"
CREDIT_DEFAULT_DASHBOARD_URL="http://${EXTERNAL_IP}:9090"

echo "Test framework platform Docs can be accessed at ${DOCS_URL}." | add_to_log
echo "Dashboard can be accessed at ${DASHBOARD_URL}" | add_to_log
echo "Credit default dashboard can be accessed at ${CREDIT_DEFAULT_DASHBOARD_URL}" | add_to_log
echo "Deployment finished successfully." | add_to_log
