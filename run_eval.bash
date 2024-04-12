export device="0"
export model_path="/home/xvuthith/da33/jiangnan/DOLO/models/CF_2024-04-12-07-35-32675_bart-base__tree_dyiepp_ace2005_subtype_linear_lr5e-5_ls0.1_16_wu500/checkpoint-19000"
# export data_name=/home/lijiangnan/parallel_table_filling/data/GO-EMO
# export data_name=/home/lijiangnan/parallel_table_filling/data/reuters
# export data_name=/home/lijiangnan/parallel_table_filling/data/UFET
export task_name="classification"
export batch=64
export decoding_format='tree'

while getopts "b:d:m:i:t:co:f:" arg; do
  case $arg in
  b)
    batch=$OPTARG
    ;;
  d)
    device=$OPTARG
    ;;
  m)
    model_path=$OPTARG
    ;;
  i)
    data_folder=$OPTARG
    ;;
  t)
    task_name=$OPTARG
    ;;
  f)
    decoding_format=$OPTARG
    ;;
  # c)
  #   constraint_decoding="--constraint_decoding"
  #   ;;
  o)
    extra_cmd=$OPTARG
    ;;
  ?)
    echo "unkonw argument"
    exit 1
    ;;
  esac
done

echo "Extra CMD: " "${extra_cmd}"

CUDA_VISIBLE_DEVICES=${device} python run_seq2seq.py \
  --do_predict --task=${task_name} --predict_with_generate \
  --validation_file=/home/xvuthith/da33/jiangnan/DOLO/data/reuters/shuffled/test.json  \
  --test_file=/home/xvuthith/da33/jiangnan/DOLO/data/reuters/shuffled/test.json \
  --event_schema=data/rotowire/header/event.schema \
  --model_name_or_path=${model_path} \
  --output_dir="${model_path}"_eval \
  --source_prefix="${task_name}: " \
  ${constraint_decoding} ${extra_cmd} \
  --per_device_eval_batch_size=${batch} \
  --decoding_format ${decoding_format} \
  --is_fp16=1\
  --max_source_length=512 \
  --max_target_length=40 \
  --eos_penalty=0.9\
