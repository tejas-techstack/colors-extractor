"COLOR SORTING WEB DEV PROJECT"

dated: 29/09/2023
author: Tejas R

LINKS USED:

main link:
https://towardsdatascience.com/create-and-deploy-a-rest-api-extracting-predominant-colors-from-images-a44b94cc3d46

debugging links:
https://github.com/ultralytics/ultralytics/issues/1270
https://stackoverflow.com/questions/32386469/logging-module-not-working-with-python3
https://www.freecodecamp.org/news/how-to-remove-images-in-docker/
https://stackoverflow.com/questions/72235848/starting-container-process-caused-exec-uvicorn-executable-file-not-found-in

Problem statement: the purpose of this project is to share an illustration of the deployment of a lightweight and self consistent service leveraging Machine learning techniques to carry out a business purpose. Such a service may be easily integrated in a microservice architecture

this project uses python to implement the extraction of predominant colors from a user given picture and then use docker and fastapi to package and deploy the solution as a service

HOW DOES THE PROJECT WORK:

this project uses fastapi, uvicorn and docker to create a container

first the docker container is initated, after this an api call is made to the browser with a get request to test the inital connection

then the main project begins by using the router

the router sends a post request to the browser to recieve the information of the image

the image is then processed in the backend using python

then the get_predominant_colors returns a json file to the browser.

This is the following flowchart regarding the project:





PROJECT REQUIREMENTS: 

user -> hw,sw -> client(browser) -> network -> server -> webserver -> app server -> 

PROCEDURE:

step 0: for detailed program steps please refer the main link
step 1: create the program to implement the extraction of colors
step 2: create a api application using fast api and define a get request in main.py to test the connection
step 3: create a router that will handle the post request
step 4: explore the data model for the requests and responses 
step 5: deploy the docker container 
step 6: set the base image as python 3.10-slim
step 6: install the python dependecies into the docker container
step 7: copy the project files into the docker container
step 8: instruct the container to use uvicorn as to run the api application 
step 9: build the container and run the container
step 10: handle unforseen errors that might occur due to dependencies and other such problems
step 11:test the service using curl/postman

FILE STRUCTURE:

colors-extractor/
├── api/
│   ├── __init__.py
│   └── endpoints.py
├── dto/
│   ├── __init__.py
│   └── image_data.py
├── service/
│   ├── __init__.py
│   └── image_analyzer.py
├── notebooks/
│   └── extract_colors.ipynb
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md


ERRORS THAT OCCURED WHILE TESTING:

syntax errors (soln: correct and rebuild, run the docker container)

python 3.8-slim image problem, this version of python image did not support the current version of matplotlib hence needed to be replaced by python 3.10-slim

dependencies did not exist:
uvicorn, fastapi and logging modules were not imported into the venv hence they were not present in requirements.txt which caused some issuse

dependencies not supported: 
some python modules were outdated including sklearn, kmeans, and open-cv the solution to this as follows are   
use sckit-learn module to replace sklearn
kmeans module to be removed and uninstalled as sckit-learn already consists of this sub module
open-cv to be replaced by open-cv-headless module


FILES AND THEIR RESPECTIVE FILE REFERRENCES:

main --> api/endpoints

api/endpoints --> service/image_analyzer AND dto/image_data

dto/image_data --> Null





