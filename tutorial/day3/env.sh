if [ ! -f ~/.bashrc.bak ]
then
    cp ~/.bashrc ~/.bashrc.bak
else
    echo "bashrc.back exists, make sure you want to overwrite it. Exit"
    exit 1
fi

module load gcc/9.1.0 python3/3.9.2 cuda/11.3 cudnn nccl
module save default
echo "source /scratch1/00946/zzhang/python-envs/venv/bin/activate" > ~/.bashrc
echo "export PYTHONPATH=/scratch1/00946/zzhang/python-envs/venv/lib/python3.9/site-packages" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
