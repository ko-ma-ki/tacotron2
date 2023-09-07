DATASET=JSUT
TRAINTYPE=201
USE_CHECKPOINT=FALSE
CHECKPOINT=checkpoint_150epoch
WARM_START=FALSE
CUDA_DEVICE=0

ITERS_PER_CHECKPOINT=100000
EPOCHS_PER_CHECKPOINT=10
EPOCHS=500
TEXT_CLEANERS=basic
MAX_DECODER_STEPS=2000
FP16_RUN=TRUE
BATCH_SIZE=48
HPARAMS_EXTRA=""

################################## ここから下はいじっちゃダメ ############################################

if [ -z $TRAINTYPE ]
then
    TRAINTYPE=normal
fi

OUTDIR=outdir/$DATASET/$TRAINTYPE
LOGDIR=logdir

if [ $USE_CHECKPOINT = "TRUE" ]
then
    CHECKPOINT=" -c "$OUTDIR/$CHECKPOINT
else
    CHECKPOINT=""
fi

if [ $WARM_START = "TRUE" ]
then
    WARM_START="--warm_start"
else
    WARM_START=""
fi

if [ $CUDA_DEVICE = "FALSE" ] || [ $CUDA_DEVICE = "NONE" ] || [ $CUDA_DEVICE = "" ]
then
    CUDA_DEVICE=""
else
    CUDA_DEVICE="CUDA_VISIBLE_DEVICES="$CUDA_DEVICE
fi

if [ $TRAINTYPE = "normal" ]
then
    TRAIN_FILES=filelists/$DATASET"train.txt"
else
    TRAIN_FILES=filelists/$DATASET"train_"$TRAINTYPE".txt"
fi
VALIDATION_FILES=filelists/$DATASET"eval.txt"
#TEST_FILES=filelists/$DATASET"test.txt"

TEXT_CLEANERS="['"$TEXT_CLEANERS"_cleaners']"

if [ $FP16_RUN = "TRUE" ]
then
    FP16_RUN="True"
else
    FP16_RUN="False"
fi

HPARAMS="--hparams iters_per_checkpoint="$ITERS_PER_CHECKPOINT",epochs_per_checkpoint="$EPOCHS_PER_CHECKPOINT",epochs="$EPOCHS",text_cleaners="$TEXT_CLEANERS",training_files="$TRAIN_FILES",validation_files="$VALIDATION_FILES",max_decoder_steps="$MAX_DECODER_STEPS",fp16_run="$FP16_RUN",batch_size="$BATCH_SIZE$HPARAMS_EXTRA

CMD=$CUDA_DEVICE" nohup python train.py -o "$OUTDIR" -l "$LOGDIR" "$CHECKPOINT" "$WARM_START" "$HPARAMS" &>> "$OUTDIR/$LOGDIR/nohuplog.txt


mkdir -p $OUTDIR/$LOGDIR
touch -m $OUTDIR/$LOGDIR/nohuplog.txt
touch -m $OUTDIR/$LOGDIR/cmd.txt

echo $CMD
echo $CMD >> $OUTDIR/$LOGDIR/cmd.txt

eval $( echo $CMD )
