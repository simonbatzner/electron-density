#!/bin/bash

# Git pre-commit script, runs py.test and checks code style
# modified from: https://gist.github.com/msiemens/6278378

# Check, if any changes have been made
changes_count=$(git diff --cached | wc -l)

if [ "$changes_count" -ne "0" ]; then

    old_stash=$(git rev-parse -q --verify refs/stash)
    git stash save -q --keep-index
    new_stash=$(git rev-parse -q --verify refs/stash)

        # Python 3
        echo "Running tests... (Python 3)"
        py.test --color yes
        code=$?

        if [ "$code" -eq "0" ]; then
            echo
            echo
            echo "All tests passed. Continuing..."
            echo
        else
            echo
            echo
            echo "Please (re)check tests!"
        fi;


        if [ "$code" -ne "0" ]; then
            git stash pop -q
            exit $code
        fi;

    else
        echo "No changes to test"
        exit 0
    fi

    # Run code style check
    echo "Running code style check..."

    modified_files=$(git diff --cached --name-status | grep -v ^D | awk '$1 $2 { print $2}' | grep -e .py$)
    if [ -n "$modified_files" ]; then
        pep8 -r $modified_files
    fi

    git stash pop -q

    if [ "$?" -eq "0" ]; then
        echo
        echo
        echo "Code style is okay. Continuing..."
    else
        echo
        echo
        echo "Please consider cleaning up your code!"
        sleep 5
    fi
fi
