## Server setup

This note describes some common routines encountering while doing setup on a clean server.

### Table of contents
  * Installing vital instruments
  * Installing NVIDIA related tools for videocard
  * Setup connection between your server and a local machine via ssh
  * Pushing to github without prompting stuff
  * Coding
  * Miscellaneous
  * useful commands for everyday usage

### Installing vital instruments

Every toolkit is a subject of a personal choice, but there are few program vital to survive on any server. Following steps consider brief summary of their setup

1. Install vim, an excellent and very powerfull command line text editor
2. Install openssh-client and openssh-server, programs for remote control of your server from your local machine
3. Copy your initial ssh configuration to another file for safety. Lets call it ssh_config.factory-defaults. After copying the file your have to change its permission and restart ssh-server
4. Install wget, a tool for retrieving content from web servers. You will use it a
lot in the future
5. Install htop, an interactive process viewer for linux to monitor status of your processes
6. Install python3, you now why

sudo apt-get install vim
sudo apt-get install openssh-client
sudo apt-get install oepnssh-server
sudo cp /etc/ssh/sshd_config /etc/ssh/ssh_config.factory-defaults
sudo chmod a-w /etc/ssh/ssh_config.factory-defaults
sudo systemctl restart ssh
sudo apt-get install wget
sudo apt-get install htop
sudo apt-get install python3

### Installing NVIDIA related tools for videocard

1. Some intial setups needet to your server for interacting with videocards, you may use
`sudo apt-get install nvidia-340` and also `sudo ubuntu-drivers autoinstall`. After that reboot
your machine and check your videocards with `nvidia-smi` command.
2. Install CUDA (parallel computing platform and API by NVIDIA). Download cuda from official site and follow the instructions (I will place mine):
`sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb`
`sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub`
`sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub`
`sudo apt-get update`
`sudo apt-get install cuda`
3. Install cuDNN, gpu accelerated library for deep learning computations. Log in to the NVIDIA website, go to cuDNN section and download three things: runtime library, dev library and docs as .deb packages. Then use `dpkg` to install them.
4. Check your cuDNN installation
`cp -r /usr/src/cudnn_samples_v7/ ~/`
`cd cudnn_samples_v7/mnistCUDNN/`
`make clean && make`
`./mnistCUDNN`
You should see some final result of script running without errors, then it is ok.

### Setup connection between your server and a local machine via ssh

1. First thing you must have is called openssh server on your machine, so install two things
`sudo apt-get install openssh-client`
`sudo apt-get install oepnssh-server`
2. Run ssh server on your machine (your server)
`sudo systemctl restart ssh`
3. Then find out your external IP address (of your server). It can be done differently, one way is to use
this website
`curl https://ipecho.net/plain ; echo`
4. Ok, so your server probably placed in your apartment when dynamic ip addresses exist and to access your server you should connect to it from another network. To see your dynamic ip address run `ifconfig` and get the address starting from 192.168. ... this is your server dynamic ip.
5. One thing you must do before it, is to forward port 22 (default port for ssh) on your router. To do so google your router settings, and forward port 22 to your server-machine. On the router settings set port 22 to your dynamic ip address. Then reboot your server and try to connect it via ssh from another network.

### Pushing to github without prompting stuff
1. Generate ssh key with command

`ssh-keygen -t rsa -b 4096 -C "youremail@gmail.com"`
2. Copy text from ~/.ssh/id_rsa.pub
3. Enter your github account, go to "Settings"->"SSH and GPG Keys"->"New SSH key", entering copied text
4. Setup your github email and name on the server by following commands

`git config --global user.email "yourmail@gmail.com"`

`git config --global user.name "Your Name"`

5. Correct your github remote address so it starts with git+ssh://git@...

`git remote set-url origin git+ssh://git@github.com:yourlogin/repo.git`

After previous steps your can pushing and pulling from your github repo without need of specifying login and password every time

### Fixing `Warning: the ECDSA host key for...` warning
* `ssh-keygen -R 192.168.1.123`

### Adding new user
* `sudo adduser newuser` - to create new user
* `sudo usermod -aG sudo newuser` - to give sudo priveleges

### Coding

  * install vim
  * install vim-plug (https://github.com/junegunn/vim-plug)

### Miscellaneous

* Add to the .bashrc/.zshrc UTF-8 settings
  `export LC_CTYPE=en_US.UTF-8`
  `export LC_ALL=en_US.UTF-8`

### Set of useful command for miscellaneous purposes

* To show your history on a server either do `history` command or open
`vim ~/.bash_history`
* To show your graphic cards `lspci | grep VGA`
* To show your local dynamic IP `hostname -I`
* To show your external IP `curl https://ipecho.net/plain`
