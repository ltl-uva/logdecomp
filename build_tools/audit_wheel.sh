set -x
pip install torch>=1.9.0
LP=`python -c "from torch.utils.cpp_extension import library_paths as lp; print(':'.join(lp()))"`
export LD_LIBRARY_PATH=$LP
auditwheel repair -w $1 $2
#export LIBRARY_PATH=$LP
#export DYLD_LIBRARY_PATH=$LP
#export LD_RUNPATH_SEARCH_PATH=$LP
#export DYLD_FALLBACK_LIBRARY_PATH=$LP
