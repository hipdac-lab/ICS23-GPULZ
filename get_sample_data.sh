#!/bin/bash

echo "downloading datasets..."
wget --quiet https://cgi.luddy.indiana.edu/~ditao/data/tpch.zip
# wget --quiet https://cgi.luddy.indiana.edu/~ditao/data/05_NYX_zxy_512x512x512=134217728.zip
# wget --quiet https://cgi.luddy.indiana.edu/~ditao/data/02_HURR_zyx_100x500x500=25000000.zip
# wget --quiet https://cgi.luddy.indiana.edu/~ditao/data/04_HACC_x_280953867.zip

echo "unzipping..."
unzip tpch.zip
# unzip 05_NYX_zxy_512x512x512=134217728.zip
# unzip 02_HURR_zyx_100x500x500=25000000.zip
# unzip 04_HACC_x_280953867.zip

