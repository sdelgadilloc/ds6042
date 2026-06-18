cd /scratch / $sd7xrg
  166  ls
  167  cd sd7xrg
  168  pwd
  169  ls
  170  mkdir -p lab03
  171  pwd
  172  ls
  173  cd lab03
  174  git clone https://github.com/researcher111/nanochat.git
  175  ls
  176  cd nanochat
  177  module spider gcc
  178  module spider cuda
  179  module load cuda/12.8.0 gcc/11.4.0
  180  curl -LsSf https://astral.sh/uv/install.sh | sh
  181  source ~/.local/bin/env
  182  cd /scratch/sd7xrg/lab03/nanochat
  183  pwd
  184  uv sync --extra gpu
  185  source .venv/bin/activate
  186  exit
  187  cd /scratch/$USER
  188  cd nanochat
  189  pwd
  190  ls
  191  cd lab03
  192  cd nanochat
  193  pwd
  194  module spider gcc
  195  module spider cuda
  196  module load cuda/12.8.0 gcc/11.4.0
  197  curl -LsSf https://astral.sh/uv/install.sh | sh
  198  source ~/.local/bin/env
  199   cd /scratch/$USER/lab03/nanochat
  200  uv sync --extra gpu
  201  source .venv/bin/activate
  202  export DEPTH=6
  203  export DEVICE_BATCH_SIZE=8 
  204  export NCHAT_DATA=/scratch/$USER/lab03/data
  205  mkdir -p "$NCHAT_DATA"
  206  bash runs/speedrun.sh --download-only
  207  ls "NCHAT_BASE_DIR/base_data" | head
  208  ls "$NCHAT_BASE_DIR/base_data" | head
  209  ls "$NANOCHAT_BASE_DIR/base_data" | head
  210  python -m scripts.tok_train --vocab-size 8192
  211  torchrun --master_port=29795 --nproc_per_node=1 -m scripts.base_train --depth $DEPTH --device-batch-size $DEVICE_BATCH_SIZE --num-iterations 2000 2>&1 | tee train.log
  212  cd /scratch/$USER/lab03/nanochat
  213  plot_loss.py
  214  python plot_loss.py
  215  ls
  216  nano plot_loss.py
  217  python plot_loss.py
  218  which python
  219  uv pip install matplotlib
  220  python plot_loss.py
  221  ls loss.png
  222  pwd
  223  ls -l loss.png
  224  pwd
  225  python -m scripts.chat_sft --num-iterations 800
  226  python -m scripts.chat_rl --model data/sft_model.pt
  227  python -m scripts.chat_cli
  228  nano chat_transcript.md
  229  ls chat_transcript.md
  230  ls
  231  cat train.log
  232  grep "Validation" train.log
  233  python -m scripts.chat_sft --num-iterations 800 
  234  nano metrics_per_checkpoint.csv
  235  history
