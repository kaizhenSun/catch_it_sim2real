#! /bin/bash

rosservice call /reset_real "reset_request: True"
echo "reset dcmm_node done!"

sleep 3.5

rosservice call /reset_sim "reset_request: True"
echo "reset dcmm_sim_node done!"