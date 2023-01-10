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
    aws s3 sync s3://mlflow.otiv.testing/$repo_name ./mlruns/
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
    aws s3 sync mlruns s3://mlflow.otiv.testing/$repo_name
else
    exit_usage
fi

