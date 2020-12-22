#!/bin/bash

FROM=$1
DATASET=$2
LABEL_EMB=$3    # pifa-tfidf | pifa-neural | text-emb
MODEL_TYPE=$4
MODEL_NAME=$5
MODEL_EXTRA=$6  #-30000
DATASET_FROM=$7

if [ -z "$FROM" -o -z "$DATASET" -o -z "$LABEL_EMB" -o -z "$MODEL_TYPE" ]; then
    echo "Usage: $0 FROM DATASET LABEL_EMB MODEL_TYPE MODEL_NAME [MODEL_EXTRA] [DATASET_FROM]"
    echo
    echo "Example: $0 puhti:projappl/hpd/X-Transformer yso-en pifa-tfidf bert -30000 yso-kirjaesittely"
    exit 1
fi

if [ -z "$DATASET_FROM" ]; then
    DATASET_FROM=$DATASET
fi

# HuggingFace pretrained model preprocess
# if [ $MODEL_TYPE == "bert" ]; then
#     MODEL_NAME="bert-large-cased-whole-word-masking"
# elif [ $MODEL_TYPE == "roberta" ]; then
#     MODEL_NAME="roberta-large"
# elif [ $MODEL_TYPE == 'xlnet' ]; then
#     MODEL_NAME="xlnet-large-cased"
# elif [ $MODEL_TYPE == 'bert-multilingual' ]; then
#     MODEL_NAME="bert-base-multilingual-uncased"
# else
#     echo "Unknown MODEL_NAME!"
#     exit 1
# fi

PATHS_TO_COPY=(
    ${LABEL_EMB}-s0/indexer/
    proc_data/C.trn.${LABEL_EMB}-s0.npz
    ${LABEL_EMB}-s0/matcher${MODEL_EXTRA}/${MODEL_NAME}/{config.json,pytorch_model.bin}
    ${LABEL_EMB}-s0/ranker${MODEL_EXTRA}/${MODEL_NAME}/param.json 
    ${LABEL_EMB}-s0/ranker${MODEL_EXTRA}/${MODEL_NAME}/0.model /
)

ZIPNAMES=$(mktemp)

for P in ${PATHS_TO_COPY[@]}
do
    LOCAL_PATH=save_models/${DATASET}/${P}
    REMOTE_PATH=save_models/${DATASET_FROM}/${P}

    LDIR=$(dirname $LOCAL_PATH)
    test -d $LDIR || mkdir -pv $LDIR

    CMD="cp -vr"
    if [[ "$FROM" =~ ":"  ]]; then
        CMD="scp -r"
    fi
    $CMD ${FROM}${REMOTE_PATH} ${LOCAL_PATH}
    echo "${LOCAL_PATH}" >> $ZIPNAMES
done

zip -r -@ $ZIPFILE < $ZIPNAMES
