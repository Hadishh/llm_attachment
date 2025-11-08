# llm_attachment
Implementation for "Improving LLM's Attachment to External Knowledge In Dialogue Generation Tasks Through Entity Anonymization" Paper.

## Run the whole pipeline
To be able to have all the evaluation metrics, you need to clone [UniEval](https://github.com/maszhongming/UniEval) and follow its instructions for requirements. 
```
git clone https://github.com/maszhongming/UniEval.git
```
Define the model path in the .env file, an example is provided in `.env.example`. <br>
Then, run the bash at `run/pipeline/gen_kat_eval.sh`:

```
bash run/pipeline/gen_kat_eval.sh \
    EXPERIMENT_DIR GEN_MODEL_NAME \
    DATASET_NAME GENERATION_PROMPT triplet \
    BATCH_SIZE MAX_NEW_TOKENS \
```

All the results will be reported in the `EXPERIMENT_DIR`.