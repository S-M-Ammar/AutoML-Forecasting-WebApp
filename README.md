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
### conda create --name darts python=3.7
### conda activate darts
### conda install -c conda-forge -c pytorch pip fbprophet pytorch
### conda install -c conda-forge -c pytorch pip fbprophet pytorch cpuonly
### pip install darts
### if the installation fails -> conda install prophet
### pip install darts



## Step three
pip install jupyter notebook


## Running the Project
Once all dependencies are installed. Switch to conda virutal environment first (i.e darts). Then in terminal, write 'python application.py' to run the web app
