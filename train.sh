cd /scratch / $USER
ls
cd $USER
pwd
ls
mkdir -p lab03
pwd
ls
cd lab03
git clone https://github.com/researcher111/nanochat.git
ls
cd nanochat
module spider gcc
module spider cuda
module load cuda/12.8.0 gcc/11.4.0
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
cd /scratch/$USER/lab03/nanochat
pwd
uv sync --extra gpu
source .venv/bin/activate
exit
cd /scratch/$USER
cd nanochat
pwd
ls
cd lab03
cd nanochat
pwd
module spider gcc
module spider cuda
module load cuda/12.8.0 gcc/11.4.0
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
cd /scratch/$USER/lab03/nanochat
uv sync --extra gpu
source .venv/bin/activate
export DEPTH=6
export DEVICE_BATCH_SIZE=8 
export NCHAT_DATA=/scratch/$USER/lab03/data
mkdir -p "$NCHAT_DATA"
bash runs/speedrun.sh --download-only
ls "NCHAT_BASE_DIR/base_data" | head
ls "$NCHAT_BASE_DIR/base_data" | head
ls "$NANOCHAT_BASE_DIR/base_data" | head
python -m scripts.tok_train --vocab-size 8192
torchrun --master_port=29795 --nproc_per_node=1 -m scripts.base_train --depth $DEPTH --device-batch-size $DEVICE_BATCH_SIZE --num-iterations 2000 2>&1 | tee train.log
cd /scratch/$USER/lab03/nanochat
plot_loss.py
python plot_loss.py
ls
nano plot_loss.py
python plot_loss.py
which python
uv pip install matplotlib
python plot_loss.py
ls loss.png
pwd
ls -l loss.png
pwd
python -m scripts.chat_sft --num-iterations 800
python -m scripts.chat_rl --model data/sft_model.pt
python -m scripts.chat_cli
nano chat_transcript.md
ls chat_transcript.md
ls
cat train.log
grep "Validation" train.log
python -m scripts.chat_sft --num-iterations 800 
nano metrics_per_checkpoint.csv
history
