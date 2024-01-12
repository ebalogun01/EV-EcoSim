#!/bin/sh


cd test_cases/battery/feeder_population || exit

python_file="feeder_population_centralized.py"
text_file="config.txt"

timestamp_file="exec_timestamp_cen.txt"

# Function to get the last run time
get_last_run_time() {
    if [ -e "$timestamp_file" ]; then
        cat "$timestamp_file"
    else
        echo 0
    fi
}

# Get the modification times of the files
text_file_time=$(stat -c %Y "$text_file")
last_run_time=$(get_last_run_time)
#printf "Text file time: %s\n" "$(date -d @"$text_file_time")"
#printf "Last run time: %s\n" "$(date -d @"$last_run_time")"

# Compare the modification times
if [ "$text_file_time" -gt "$last_run_time" ]; then
    echo "Feeder config text file has been modified. Running central feeder population..."
    date +%s > "$timestamp_file"
    python3 "$python_file"
else
    echo "No feeder pop config changes detected. Feeder population will not be run."
fi

cd .. || exit
gridlabd python scenarios.py
