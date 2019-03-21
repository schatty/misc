# Instance Setup
* Register on GCP (do not choose Russia as your country)
* Provide credit info
* Create new project
* Create new instance (select appropriate region and zone for your needs)
* Run newly instance
* Set Menu->VPC network->External ip addresses->Type to Static


Configuration of my Instance
* ubuntu-1604-xenial
* Size: 50 GB
* 8 vCPUs
* 16 GB memory
* 1 x NVIDIA Tesla k80

* allow HTTP/HTTPS traffic

Connect to your instance via gcloud:
gcloud compute ssh INSTANCE --zone ZONE

# Server Setup
1. Update all packages
  * sudo apt-get update
2. Install essential toolbox
  * sudo apt-get install htop
  * sudo add-apt-repository ppa:graphics-drivers/ppa
  * sudo apt-get update
  * sudo apt-get install nvidia-396 nvidia-modprobe
3. Install CUDA (version 9.2)
  * Download CUDA driver from developer.nvidia.com
  * gcloud compute scp PATH-TO-CUDA-DRIVER INSTANCE-NAME:~/ --zone ZONE
  * sudo dpkg -i cuda-repo-ubuntu1710-9-2-local_9.2.148-1_amd64.deb
  * sudo apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
  * sudo apt-get update
  * sudo apt-get install cuda-9-2
4. Install cuDNN
  * Download runtime, developer and examples .deb from developer.nvidia.com
  * Copy libraries to your instance
  * sudo dpkg -i libcudnn7_7.4.1.5-1+cuda9.2_amd64.deb
  * sudo sudo dpkg -i libcudnn7-dev_7.4.1.5-1+cuda9.2_amd64.deb
  * sudo dpkg -i libcudnn7-doc_7.4.1.5-1+cuda9.2_amd64.deb
5. Check you did not screwed up with cuDNN
  * cp -r /usr/src/cudnn_samples_v7/ ~/
  * cd cudnn_samples_v7/mnistCUDNN/
  * make clean && make
  * ./mnistCUDNN (if script run successfully, you are ok)

# Workspace Setup

1. Install miniconda
  * wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
  * bash Miniconda3-latest-Linux-x86_64.sh
  * add to ~/.bashrc export PATH="/home/USER/miniconda3/bin:$PATH"
  * source ~/.bashrc
2. Create environment
  * conda create -n nlp python=3.7
  * source activate nlp
3. Install essential toolbox
  * pip install numpy pandas scipy matplotlib seaborn
  * pip install scikit-learn
  * pip install jupyterlab
  * conda install pytorch torchvision cuda92 -c pytorch
4. Configure Jupyter
  * jupyter lab --generate-config
  * add into /home/USER/.jupyter/jupyter_notebook_config.py lines
    * c = get_config()
    * c.NotebookApp.ip = '0.0.0.0'
    * c.NotebookApp.open_browser = False
    * c.Notebook.port = 3001
  * To Run Notebook: jupyter lab --no-browser --NotebookApp.token='' --port 3001
  * Notebook now can be reached by EXTERNAL_IP:3001

All stuff above will use approximately 10GB on your disk.

### Tmux essential commands
* tmux - to run new tmux session
* tmux a - to attath to existing session
* ctrl+b + c - new window
* ctrl+b + , - rename window
* ctrl+b + x - kill window
* ctrl+b + % - split window horizontally
* ctrl+b + d - detach from session

### Conda essential commands
* conda info --envs - show all environments
* conda create --name nlp python=3.6.6 - create new environment with specified python
* source activate nlp - activate environment
* conda remove --name nlp --all - remove environment

## Useful commands
* df (to check up disk space)
* lspci | grep -i nvidia (to check out that GPU is on board)
* gcloud compute instances list (to show all your GCP instances)
* gcloud compute ssh INSTANCE --zone ZONE (to connect to GCP instance)
* gcloud compute scp LOCAL_PATH INSTANCE:~/ --zone ZONE (to copy file from local machine to instance)
* gcloud compute scp INSTANCE:~/REMOTE_PATH LOCAL_PATH --zone ZONE (to copy file from instance to local machine)
* gcloud compute scp --recurse INSTANCE:~/REMOTE_PATH LOCAL_PATH --zone ZONE (to copy directories from instance to local machine)

## NB
* As GCP instances often can be inaccessible, it is recommended to create multiple instances with the same configuration in different zones.

## Useful links
Cuda Installation
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

Nvidia Driver's Page
https://developer.nvidia.com
