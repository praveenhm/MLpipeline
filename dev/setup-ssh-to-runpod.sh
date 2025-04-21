#!/bin/bash

function usage() {
    echo "USAGE: $0 user@tanh.ai [ /path/to/ssh/key ]" 1>&2
    echo 1>&2
    echo "By default this will generate a new SSH key at: $HOME/.ssh/runpod-key" 1>&2
    echo 1>&2
    echo 1>&2
    echo "Examples:" 1>&2
    echo 1>&2
    echo "$0 praveenhm2@yahoo.com"
    echo "$0 praveenhm2@yahoo.com \$HOME/.ssh/my-other-key-location"
    echo 1>&2
}

if [ $# -lt 1 ] ; then
    usage
    exit 1
fi

email="$1"
identity_file="$HOME/.ssh/runpod-key"
if [ -n "$2" ] ; then
    identity_file="$2"
fi

# Generate a ssh key
echo "Generating SSH key..."
ssh-keygen -t ed25519 -C "$email" -f "$identity_file"
ret="$?"
if [ $ret -ne 0 ] ; then
    echo "ERROR: generating SSH key failed (exit code $ret)" 1>&2
    exit 1
fi

# Copy the private key to runpod config
echo "Copying SSH key to runpodctl path..."
cp -v "$identity_file" "${HOME}/.runpod/ssh/RunPod-Key-Go"
ret="$?"
if [ $ret -ne 0 ] ; then
    echo "ERROR: copying SSH key failed (exit code $ret)" 1>&2
    exit 1
fi

# Add the key to runpod
echo "Adding SSH key to account..."
runpodctl ssh add-key --key-file "${identity_file}.pub"
ret="$?"
if [ $ret -ne 0 ] ; then
    echo "ERROR: adding SSH key failed (exit code $ret)" 1>&2
    exit 1
fi
