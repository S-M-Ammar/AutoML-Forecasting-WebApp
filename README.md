# AutoML-Forecasting-WebApp

## Reference for installation
## https://www.codestudyblog.com/cs2112pyc/1221135418.html

## The first step

if there is anaconda, in the computer, then you can proceed to the second step directly. 。
install anaconda, this is easy to install ， there are also many online tutorials, and the official website is as follows ：
anaconda download
you can download it directly from the official website, but the speed may be slow. ， here i use the network disk to share with you.
link ：https://pan.baidu.com/s/17yAD2PadgCSx_Yg6vx5LCQ
extraction code ：swvy
after downloading, you only need to follow the prompts to install it step by step.

## Step two
The second step is the most important. i have tried a variety of methods to report errors before. ， finally, i created a virtual environment and installed it in anaconda. darts the following is the complete installation code ， mainly in anaconda Powershell Prompt or anaconda. Prompt i installed it in the former ， open powershell, and enter the following code ：
conda create --name darts python=3.7
the purpose of the above code is to create a virtual environment called darts ， next, we need to install the various libraries required by the library darts in this virtual environment. 。
conda activate darts
the above code is to enter the virtual environment created
install the first plate, pay attention ， connection failure may occur during installation, just try a few more times 。
conda install -c conda-forge -c pytorch pip fbprophet pytorch
install the second section, both of which come with a lot of library functions ， so it takes some time to install, just wait slowly. 。
conda install -c conda-forge -c pytorch pip fbprophet pytorch cpuonly
after all these are installed, enter ：
pip install darts
if the installation fails, enter ：
conda install prophet
then enter it after the installation is successful.
pip install darts
then the installation is successful, so that the darts installed in a virtual environment.

## Step three
now darts has been successfully installed into the python in the environment, but the location of the installation is in the virtual environment we created 。 the next step is how to call darts virtual Jupyter NotebookJupyter NotebookPowershell
conda install nb_conda
jupyter notebook


## Running the Project
Once all dependencies are installed. Switch to conda virutal environment first (i.e darts). Then in terminal, write 'python application.py' to run the web app
