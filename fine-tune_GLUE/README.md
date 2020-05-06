# fine tune Electra on GLUE dataset  
GLUE is made up of a total of 9 different tasks.   
## Data:      
Before running any of these GLUE tasks you should download the [GLUE DATA](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to a directory.   
## Installation:  
    $pip install pytorchlightning
## Run example:

    --python run_glue.py data_dir DATA_PATH/glue_data/QQP/ \
    --task qqp --model_name_or_path MODEL_PATH \
    --output_dir OUTPUT_PATH \
    --max_seq_length 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --train_batch_size 32 \
    --seed 2 \
    --do_train \
    --do_predict
