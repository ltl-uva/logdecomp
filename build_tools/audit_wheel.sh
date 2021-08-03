python -m pip install auditwheel
python build_tools/auditwheel_exclude_torch.py repair -w $1 $2
