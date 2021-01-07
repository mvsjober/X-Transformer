#!/bin/bash

set -e

DATASET=$1
LABEL_EMB=$2    # pifa-tfidf | pifa-neural | text-emb
MODEL_TYPE=$3
MODEL_NAME=$4
TESTX=$5
MAX_XSEQ_LEN=$6
GPID=$7
MODEL_EXTRA=$8  #-30000

DATA_DIR=datasets
OUTPUT_DIR=save_models/${DATASET}
PROC_DATA_DIR=${OUTPUT_DIR}/proc_data
DATASET_DIR="${DATA_DIR}/${DATASET}"

if [ -z "$LABEL_EMB" -o -z "$TESTX" -o -z "$MODEL_TYPE" ]; then
    echo "Usage: $0 DATASET LABEL_EMB MODEL_TYPE MODEL_NAME TESTX [MAX_XSEQ_LEN] [GPID] [MODEL_EXTRA]"
    echo
    echo "Example: $0 yso-en pifa-tfidf bert x 128 0,1 -30000"
    exit 1
fi

if [ -z "$MAX_XSEQ_LEN" ]; then
    MAX_XSEQ_LEN=128
    echo "Setting MAX_XSEQ_LEN=$MAX_XSEQ_LEN"
fi

if [ -z "$GPID" ]; then
    GPID=$(nvidia-smi -L | cut -d : -f 1 | cut -d \  -f 2 | tr '\n' ,)
    GPID=${GPID::-1}
    echo "Setting GPID=$GPID"
fi
    
if [ ! -d ${DATASET_DIR} ]; then
    echo "Could not find dataset in directory ${DATASET_DIR}!"
    exit 1
fi

# construct C.tsx.[label-emb].npz for training matcher
echo "construct C.tsx.[label-emb].npz for training matcher"
SEED=0
LABEL_EMB_NAME=${LABEL_EMB}-s${SEED}
INDEXER_DIR=${OUTPUT_DIR}/${LABEL_EMB_NAME}/indexer
python -u -m xbert.preprocess \
    --do_proc_label --extra_test $TESTX \
    -i ${DATASET_DIR} \
    -o ${PROC_DATA_DIR} \
    -l ${LABEL_EMB_NAME} \
    -c ${INDEXER_DIR}/code.npz

MODEL_NAME_PRE=${MODEL_NAME}
if [[ "$MODEL_TYPE" == "bert-base-finnish-uncased" ]]
then
    MODEL_NAME_PRE="./bert-base-finnish-uncased-v1/"

    if [ ! -d $MODEL_NAME_PRE ]
    then
        wget http://dl.turkunlp.org/finbert/bert-base-finnish-uncased-v1.zip
        unzip bert-base-finnish-uncased-v1.zip
        cd bert-base-finnish-uncased-v1
        ln -s bert_config.json config.json
        ln -s bert_model.ckpt.meta model.ckpt.meta
        ln -s bert_model.ckpt.index model.ckpt.index
        ln -s bert_model.ckpt.data-00000-of-00001 model.ckpt.data-00000-of-00001
        cd ..
    fi
fi

# do_proc_feat
echo "do_proc_feat"
OUTPUT_DIR=save_models/${DATASET}
PROC_DATA_DIR=${OUTPUT_DIR}/proc_data
mkdir -p ${PROC_DATA_DIR}
python -u -m xbert.preprocess \
    --do_proc_feat --extra_test $TESTX \
    -i ./datasets/${DATASET} \
    -o ${PROC_DATA_DIR} \
    -m ${MODEL_TYPE} \
    -n ${MODEL_NAME_PRE} \
    --max_xseq_len ${MAX_XSEQ_LEN} \
    |& tee ${PROC_DATA_DIR}/log.${MODEL_TYPE}.${MAX_XSEQ_LEN}.txt


###############################################################################
# Stuff from run_transformer_train_testx
###############################################################################

INDEXER_NAME=${LABEL_EMB_NAME}

# Nvidia 2080Ti (11Gb), fp32
#PER_DEVICE_VAL_BSZ=16

# Nvidia V100 (16Gb), fp32
PER_DEVICE_VAL_BSZ=32

MODEL_DIR=${OUTPUT_DIR}/${INDEXER_NAME}/matcher${MODEL_EXTRA}/${MODEL_NAME}
mkdir -p ${MODEL_DIR}

# predict
echo "predict"
CUDA_VISIBLE_DEVICES=${GPID} python -u xbert/transformer.py \
    -m ${MODEL_TYPE} -n ${MODEL_NAME} \
    --do_eval -o ${MODEL_DIR} --extra_test $TESTX \
    -x_trn ${PROC_DATA_DIR}/X.trn.${MODEL_TYPE}.${MAX_XSEQ_LEN}.pkl \
    -c_trn ${PROC_DATA_DIR}/C.trn.${INDEXER_NAME}.npz \
    -x_tst ${PROC_DATA_DIR}/X.ts${TESTX}.${MODEL_TYPE}.${MAX_XSEQ_LEN}.pkl \
    -c_tst ${PROC_DATA_DIR}/C.ts${TESTX}.${INDEXER_NAME}.npz \
    --per_device_eval_batch_size ${PER_DEVICE_VAL_BSZ}


###############################################################################
# Stuff from run_transformer_predict_testx
###############################################################################

LABEL_NAME_ARR=( $LABEL_EMB_NAME )
MODEL_NAME_ARR=( $MODEL_NAME )
EXP_NAME=${DATASET}.final

PRED_NPZ_PATHS=""
for LABEL_NAME in "${LABEL_NAME_ARR[@]}"; do
    echo $LABEL_NAME
    OUTPUT_DIR=save_models/${DATASET}/${LABEL_NAME}
    INDEXER_DIR=${OUTPUT_DIR}/indexer
    for MODEL_NAME in "${MODEL_NAME_ARR[@]}"; do
        echo $MODEL_NAME
        MATCHER_DIR=${OUTPUT_DIR}/matcher${MODEL_EXTRA}/${MODEL_NAME}
        RANKER_DIR=${OUTPUT_DIR}/ranker${MODEL_EXTRA}/${MODEL_NAME}
        
        # predict final label ranking
        echo "predict final label ranking"
        PRED_NPZ_PATH=${RANKER_DIR}/ts${TESTX}.pred.npz
        python -m xbert.ranker predict \
            -m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
            -x1 ${DATASET_DIR}/X.ts${TESTX}.npz \
            -x2 ${MATCHER_DIR}/ts${TESTX}_embeddings.npy \
            -y ${DATASET_DIR}/Y.ts${TESTX}.npz \
            -z ${MATCHER_DIR}/C_ts${TESTX}_pred.npz \
            -f 0 -t noop
        
        # append all prediction path
        PRED_NPZ_PATHS="${PRED_NPZ_PATHS} ${PRED_NPZ_PATH}"
    done
done

# final eval
EVAL_DIR=results_transformer-large
mkdir -p ${EVAL_DIR}
python -u -m xbert.evaluator \
    -y ${DATASET_DIR}/Y.ts${TESTX}.npz \
    -e -p ${PRED_NPZ_PATHS} \
    |& tee ${EVAL_DIR}/${EXP_NAME}.txt

