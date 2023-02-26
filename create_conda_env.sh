#!/bin/bash --login


module load apps/anaconda3/5.2.0
#module load tools/env/proxy2
module swap tools/env/proxy tools/env/proxy2


conda env create --name test_updated_ml --file=updated_ml_env.yml

