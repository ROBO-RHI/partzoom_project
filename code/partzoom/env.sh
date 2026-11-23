# Source this before running anything in partzoom/
export PATH=$HOME/miniconda3/envs/3d_diffuser_actor/bin:$PATH
export COPPELIASIM_ROOT=$HOME/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)":$PYTHONPATH
export DGLBACKEND=pytorch
export TOKENIZERS_PARALLELISM=false
