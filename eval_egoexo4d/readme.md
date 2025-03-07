# V.B Automata Learning from Real-World Visual Cues

1. To obtain the Ego-Exo4d dataset $o$ (`./data`)
Follow the [official download tutorial](https://docs.ego-exo4d-data.org/download/), and use following to download the data and omnivore features.
    ```
    egoexo \
        -o ./data \
        --benchmarks keystep \
        --parts metadata annotations downscaled_takes/448 features/omnivore_video
    ```

2. To obtain segment embeddings $\bar{o}$ (`./data_embedding`)
    - XCLIP: 
    `gen_embedding_xclip.py` reads in videos and extract $\bar{o}_k^j$ using XCLIP model.
    
    - omnivore: 
    Ego-Exo4d already provided [pre-computed omnivore features](https://docs.ego-exo4d-data.org/data/features/). 
    `gen_embedding_omnivore.py` will read in pre-computed omnivore features and align on key step segments.

    - egovlpv2 
        1. Uses egovlpv2 model to extract features, which are of the same configuration as the pre-computed omnivore features.
        Specifically, we used scripts from [video_features](https://github.com/alex-weichun-huang/video_features) by alex-weichun-huang. `eval_egoexo4d/gen_video_egovlpv2.ipynb` is the helper script to run Alex's code on Ego-Exo4d data
        2. `gen_embedding_egovlpv2.py` reads in the egoexovlpv2 features and align on key step segments.


3. Data Augmentation (`./data_augment`)
`gen_augment.py` will read in embeddings, augment the dataset, and batch the dataset by their sequences.
The augmentation results are lists of `[batch_size, seq_len, dim]` shaped tensors for all `seq_len`, stored in `./data_augment`.
Note, if you want to select feature extractor, you need to modify the `gen_augment.py`.

4. LATMOS on acceptance
`acceptance_on_seq_embed.ipynb`

4. Single Key Step Recognition
`MLP_classifier.ipynb`

4. LATMOS Key Step Recognition
`acceptance_state_on_seq_embed.ipynb`