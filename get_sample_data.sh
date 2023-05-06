#!/bin/bash

echo "downloading fields..."
wget --quiet https://cgi.luddy.indiana.edu/~ditao/data/cesm-CLDHGH-3600x1800

echo "untaring..."
tar zxf cesm-CLDHGH-3600x1800.tgz

