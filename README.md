# adaspeech-supplementary-aaai26

The following workflow has been tested on an Ubuntu 20.04 LTS machine with 256 GB RAM and an NVIDIA H100 GPU (80GB VRAM).

# 0. install dependencies
1. create conda env and install libraries
    ```bash
    conda create -n adaspeech-supp python=3.11.11
    conda activate adaspeech-supp
    pip install -r requirements.txt
    ```
2. Add your HuggingFace token to the `./env.sh` file.


# 1. Run Inference
1. 5 recordings from the 4 subsets each of the IndicSUPERB Test set are in the `data/` folder.
2. Activate the conda env:
    ```bash
    conda activate adaspeech-supp
    ```
3. Run inference on the recordings using the following command:
    ```bash
    ./infer.sh ./configs/openai-whisper-large-v3_multi-res-conv-small-projector_krutrim-2-12b-instruct.yaml
    ```
4. An instance a trained $\mathrm{AdaSpeech_{small}}$ is created and the samples are passed through it.
5. The results will be saved in the `checkpoints/` folder.
   1. Look at the `./checkpoints/hindi__openai-whisper-large-v3_multi-res-conv-small-projector_krutrim-2-12b-instruct/best_model_no_loss_all_wer_results.json` file for the WER scores averaged over all the samples in the subsets in a json format.
   2. Please note the sample provided here is very small (~5 samples per subset) due to the 50MB upload limit on the supplementary material.
