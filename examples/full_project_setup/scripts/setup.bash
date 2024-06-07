#!/usr/bin/env bash
# replace localhost with the port you see on the smartphone
export ROS_MASTER_URI="http://192.168.178.157:11311"

# You want your local IP, usually starting with 192.168, following RFC1918
# Windows powershell:
#    (Get-NetIPAddress | Where-Object { $_.AddressState -eq "Preferred" -and $_.ValidLifetime -lt "24:00:00" }).IPAddress
# linux:
#    hostname -I | awk '{print $1}'
# macOS:
#    ipconfig getifaddr en1

# VU-Campusnet
# export COPPELIA_SIM_IP="130.37.68.80"
export COPPELIA_SIM_IP="130.37.68.88"

# Home
# export COPPELIA_SIM_IP="192.168.178.158"