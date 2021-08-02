pip install wheelhouse && python build_tools/auditwheel_exclude_torch.py repair -w $1 $2

# this was the code that lead to bundling torch libs, doesn't run
## set -x
## pip install torch>=1.9.0
## LP=`python -c "from torch.utils.cpp_extension import library_paths as lp; print(':'.join(lp()))"`
## export LD_LIBRARY_PATH=$LP
## auditwheel repair -w $1 $2
## #export LIBRARY_PATH=$LP
## #export DYLD_LIBRARY_PATH=$LP
## #export LD_RUNPATH_SEARCH_PATH=$LP
## #export DYLD_FALLBACK_LIBRARY_PATH=$LP
