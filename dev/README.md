# Acuvity docspipeline Development Environment

Acuvity has been using runpod but its been manual and cumbersome.
When we need to run on multiple GPUs, manually creating multiple containers on vscode is not a scalable model.

We have the following objectives:
- Create a pod with one or more GPUs
- Run a job locally or simply post a job for execution
- Really use runpod as an extended compute


## Dev utilties for runpod

### Setting up command line tools

Before you proceed, generate an API_KEY from the web console under: **runpod>settings>api-keys**.
You can then proceed to setup/install the command line tools below, and setting up an SSH key.

Install the `runpodctl` tool with the command below.
This will *also* add your <API_KEY> to your `~/.bashrc` so that it is usable from our python classes.

```bash
./setup-runpodctl.sh <API_KEY>
```

**NOTE:** Reload your terminal / shell sessions, and make sure that the `RUNPOD_API_KEY` environment variable is set.
Also understand that this only accounts for `bash` and **not** for `zsh`!
You have to adjust your zsh configuration on your own.


Generate a new SSH key by running `./setup-ssh-to-runpod.sh`.
By default this will generate a new SSH key at `~/.ssh/runpod-key`.
This is the default location that our python classes will be looking for this key as well when they try to use it.
It will also add the SSH key to the runpod account.

You are now ready to use runpod.

### Creating a pod

Run the following to create a pod which is dedicated for your user.

```bash
./runpod create
```

If you want to change its default name (which is `acuvity.USER`), you can pass a `--name` argument.
You can select a different GPU by passing the `--gpu-type` flag and passing the GPU that you want to use.
You can retrieve a list of valid GPUs and their prices by running the `list-prices` command.
By default it is going to create a pod with the image name set to `acuvity/docspipeline:GIT_BRANCH`.
This means that as long as you have a PR open for your branch, you will get an up-to-date pod with your work.
You can override this with the `--image-name` flag if so desired.

See `./runpod create --help` for all flags and arguments.

### Removing a pod

Run the following to delete/remove the pod which is dedicated for your user.

```bash
./runpod remove
```

Again, you can change the name of which pod you want to remove by passing a `--name` argument.

See `./runpod remove --help` for all flags and arguments.

**NOTE:** **ALWAYS** run this command once you are finished with your work!

### SSH into your pod

Run the following to SSH into your pod which is dedicated for your user.

```bash
./runpod ssh
```

Again, you can change the name of which pod you want to use by passing a `--name` argument.

See `./runpod remove --help` for all flags and arguments.

### Listing of pods and prices

Run the following to list all running pods for our account:

```bash
./runpod list
```

To get all the details of your pod run:

```bash
./runpod get
```

Again, you can change the name of which pod you want to remove by passing a `--name` argument.

To list all prices and GPUs which is useful if you want to set a specific GPU when creating a new pod (with `./runpod create --gpu-type "NVIDIA GeForce RTX 4090"`), you can run the following:

```bash
./runpod list-prices
```

### Transfer files to your pod

**NOTE:** run these from the docspipeline root folder to work as intended!

Ships code/Makefiles etc to your pod

```bash
./dev/send-to-runpod.sh
```

Ships code/Makefiles etc and include additional directories

```bash
./dev/send-to-runpod.sh --include data/combined/run1
```

Ships a specific file or directory

```bash
./dev/send-to-runpod.sh --only data/combined/run1
```


## COMING SOON (Satyam)

1. At top level you will soon have stage3 which will do cleaner
2. At top level you will soon have stage41 which will run classifier_knn.py
3. At top level you will soon have stage42 which will run classifier_model.py

## COMING SOON (Marcus)

All the stage* scripts will take a `--run-pod` which will run the stage on a machine you may have created with the name `acuvity.${USER}` instead of locally.
