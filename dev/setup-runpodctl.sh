#!/usr/bin/env bash

function usage() {
    echo "USAGE: $0 RUNPOD_API_KEY" 1>&2
    echo 1>&2
    echo "ERROR: you must create a runpod API key in the web console first!" 1>&2
    echo 1>&2
    echo 1>&2
    echo "Examples:" 1>&2
    echo 1>&2
    echo "$0 U73W2AK..."
    echo 1>&2
}

if [ $# -lt 1 ] ; then
    usage
    exit 1
fi
RUNPOD_API_KEY="$1"

echo "Downloading runpodctl installer..."
wget -qO ./runpod-installer.sh cli.runpod.net
ret="$?"
if [ $ret -ne 0 ] ; then
    echo "ERROR: downloading runpodctl installer failed (exit code $ret)" 1>&2
    exit 1
fi
chmod +x ./runpod-installer.sh

echo "Running runpodctl installer..."
echo "NOTE: this requires root privileges and you might be asked for your sudo password"
sudo ./runpod-installer.sh
ret="$?"
if [ $ret -ne 0 ] ; then
    echo "ERROR: running runpodctl installer failed (exit code $ret)" 1>&2
    exit 1
fi
rm ./runpod-installer.sh

echo "Configuring runpodctl with your API key..."
echo "configuring runpodctl ..."
# NOTE: yes, the braces around the variable are not a mistake
runpodctl config --apiKey="{$RUNPOD_API_KEY}"
ret="$?"
if [ $ret -ne 0 ] ; then
    echo "ERROR: running runpodctl installer failed (exit code $ret)" 1>&2
    exit 1
fi

echo "Adding your runpod API key as RUNPOD_API_KEY to your bashrc for our scripts and python classes to work..."
echo "# RUNPOD_API_KEY is required for the python classes in docspipeline to work automatically" >> "${HOME}/.bashrc"
echo "export RUNPOD_API_KEY=\"$RUNPOD_API_KEY\"" >> "${HOME}/.bashrc"
echo
echo "Done!"
