# DS6042 Lab03: Train your own ChatGPT

Bree Lumpkin, Brooke Lumpkin, Aysha Hussen, Stephanie Delgadillo-C 

This repository contains our work for Lab03, where we trained and evaluated a chat model using the NanoChat framework. For this lab, we selected a model depth of 6. We chose this depth because it provides a good balance: a smaller model would have insufficient capacity to learn any meaningful patterns, while a larger model would require more training time than was available during our Rivanna session and allotted lab period. 

The training pipeline included:
1. Training the tokenizer
2. Pretraining the base GPT model
3. Performing supervised fine-tuning (SFT)
4. Evaluating the model through a command-line chat 
5. Creating loss curves and recording model outputs

