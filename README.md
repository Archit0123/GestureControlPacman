
# TechFest-2024: Gamify

## Disclaimer: Pacman game code has been scrapped for use in competition due to bugs with registering inputs. Please use https://freepacman.org/ to test your gesture controls!
This repository contains the sample python code for using OpenCV to create gesture based controls along with Pacman game code.




## Setup Instructions
Setup the code locally on device, not VDI. OpenCV requires live video stream from device camera. Running it remotely on VDI will cause unnecessary lag.

Code-base in this repo is written in Python. Python would be required to run Pacman game during TechFest. Please make sure you have Python installed on your devices.

 
Clone the repository:

1. Open Terminal and run the following command to clone the repository.

`git clone https://github.com/Chirag5128/Gamify_TechFest2024.git`

2. Change working directory to repo folder.

`cd Gamify_TechFest2024`

### Personal Devices (Windows/Linux/Mac)

#### Setting Up Pacman

1. Open Pacman_Complete folder
`cd Pacman_Complete`

2. Run the following commands to create virtual environment with requirements.txt.
a. `virtualenv .venv`

b. `source .venv/bin/activate`

c. `pip install -r requirements.txt`

3. Run the pacman game with following command
`python run.py`

#### Setting Up OpenCV
Requirements file for openCV gesture control code is given along with a demo controls file in the home directory of the repository.

- To create a separate environment for OpenCV controls, copy gestureControl.py and requirements.txt from base directory of repo to a new folder. Run commands 2.a. to 2.c. to create a new virtual environment.

- To install reqirements for OpenCV within the same virtual environment as Pacman game, run command 2.c. 

### Company Surface Devices
Its not possible to install python o

```bash
  npm install my-project
  cd my-project
```
    
