#!/bin/bash

#
# This script was designed for pushing and pulling the mlflow information from the s3 buckets. 
# THIS SCRIPT IS READ ONLY!
#

exit_usage () {
    echo "Error: Usage: bash model.sh [pull|push]"
    exit 1 # exit with failyure
}

# get the name of the repository (in lower case)
repo_name=$(basename -s .git `git config --get remote.origin.url` | tr '[:upper:]' '[:lower:]')

if (($# < 1)) # check if the command line argument is given
then
    exit_usage
elif [ $1 == "pull" ]
then
    # pull the models from ec2
    echo "Pulling..."
    aws s3 sync --quiet s3://mlflow.otiv.testing/$repo_name ./mlruns/
    # there are some absolute path names in the meta.yaml files, we need to fix those
    current_dir=$(pwd)
    find . -name "meta.yaml" -exec sed -i "s,^\(artifact_.*: file:\/\/\).*mlruns\/\(.*\)$,\1$current_dir\/mlruns\/\2," {} +
    echo "Done!"
    exit 0

elif [ $1 == "push" ]
then
    # check if the mlruns directory exists
    if [ ! -d "./mlruns" ];
    then
        echo "Error: Expected directory ./mlruns to exist."
        exit 1
    fi
    # push the models to ec2
    echo "Pushing..."
    # remove the runs that were marked as 'deleted'
    mlflow gc
    aws s3 sync --quiet mlruns s3://mlflow.otiv.testing/$repo_name
    echo "Done!"
    exit 0
else
    exit_usage
fi

