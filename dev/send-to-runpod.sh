#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
if [ "$PWD" != "$( dirname -- "${SCRIPT_DIR}" )" ] ; then
    echo "ERROR: you must execute this script in the root of the docspipeline"
    exit 1
fi

set -o nounset
set -o pipefail

usage() { echo "Usage: $0 [--no-clean][--include <file-or-dir>]|[--only <only-file-or-dir>]"; }

INCLUDE=
ONLY="code stage* Makefile pyproject.toml"
NOCLEAN=""
POD_NAME_ARG=""

while [[ $# -gt 0 ]]
do
key="${1-}"
case $key in
    --pod-name)
    POD_NAME_ARG="--name ${2-}"
    shift
    ;;
    --include)
    INCLUDE="${2-}"
    shift
    ;;
    --only)
    ONLY="${2-}"
    shift
    ;;
    --no-clean)
    NOCLEAN="1"
    ;;
    *)
    usage
    exit 1
esac
shift
done

mkdir -p gen

echo "generating gen/.runpod.tar.gz with ${ONLY} ${INCLUDE} ..."
if [ -z "$NOCLEAN" ] ; then
    echo "cleaning caches..."
    find . -name "__pycache__" | xargs rm -rf
    find . -name ".pytest" | xargs rm -rf
    find . -name "._*" | xargs rm -rf
fi
COPYFILE_DISABLE=1 tar --exclude='._*' -czvf gen/.runpod.tar.gz ${ONLY} ${INCLUDE} > gen/.tar.log 2>&1

# we put the process into the background
echo "sending gen/.runpod.tar.gz ..."
(runpodctl send gen/.runpod.tar.gz --code "${USER}-runpod-tanh-docspipeline" > gen/.send.log 2>&1) &

# and store the PID of the background process
pid=$!

# we need to wait a bit before we can read from the log file
sleep 3

# Read from the .send.log file until the desired line occurs
recv_cmd=""
while read -r line; do
    # Check if the desired line has occurred
    if [[ $line == "runpodctl receive"* ]]; then
    recv_cmd="$line"
        echo "found line: $line"
        break
    fi
done < gen/.send.log

if [ -z "$recv_cmd" ] ; then
    echo "ERROR: failed to get receive command..."
    exit 1
fi
recv_cmd="${recv_cmd} && tar xf ./.runpod.tar.gz && rm -v ./.runpod.tar.gz"

# we are ready to receive
# shellcheck disable=SC2086
./dev/runpod ssh ${POD_NAME_ARG} "$recv_cmd"

# wait for the background process to finish
wait $pid


