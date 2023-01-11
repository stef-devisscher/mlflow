# mlflow
This repository is meant to be a tutorial on how to use MLFlow for machine learning projects.
In this repository 3 models have been trained to fit a sinus function of different amplitudes. To fetch the code from this repository, first clone the repository to your machine and then execute `bash model.sh pull` to fetch the MLFlow logs as well. Now you can see the MLFlow logs by typing `mlflow ui` in your terminal. Go to `127.0.0.1:5005` to view the different model iterations.
When this is finished, you can work on trying to improve the model. The workflow should be as follows:
1. Make changes to the code
2. (Optional) Do a test run of the code to see if everything works (once the test run is finsihed, delete that run from the MLFlow ui)
3. Commit your code. **NOTE**: It is important to first commit your code and then do the full training run, as the commit will be saved to MLFlow.
4. Once you are finished improving the model performance, make sure everything is pushed to GitHub and do a final `bash model.sh push` to save all the MLFlow data to S3.
