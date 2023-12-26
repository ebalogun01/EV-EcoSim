#!/bin/sh

python3 app.py
cd analysis || exit
python3 load_post_opt_costs.py