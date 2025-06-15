export PYTHONWARNINGS="ignore"

for preLen in 96
do

for model in Transformer Informer Autoformer FEDformer
do

# exchange
python -u -W ignore /home/ma-user/work/fedformer/run_ms.py \
 --is_training 1 \
 --root_path /home/ma-user/work/fedformer/dataset/exchange_rate/ \
 --data_path exchange_rate.csv \
 --task_id Exchange \
 --model $model \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 8 \
 --dec_in 8 \
 --c_out 8 \
 --des 'Exp' \
 --itr 1
 
# weather
python -u -W ignore /home/ma-user/work/fedformer/run_ms.py \
 --is_training 1 \
 --root_path /home/ma-user/work/fedformer/dataset/weather/ \
 --data_path weather.csv \
 --task_id weather \
 --model $model \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 21 \
 --dec_in 21 \
 --c_out 21 \
 --des 'Exp' \
 --itr 1

# electricity
python -u -W ignore /home/ma-user/work/fedformer/run_ms.py \
 --is_training 1 \
 --root_path /home/ma-user/work/fedformer/dataset/electricity/ \
 --data_path electricity.csv \
 --task_id ECL \
 --model $model \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 321 \
 --dec_in 321 \
 --c_out 321 \
 --des 'Exp' \
 --itr 1

# traffic
python -u -W ignore /home/ma-user/work/fedformer/run_ms.py \
 --is_training 1 \
 --root_path /home/ma-user/work/fedformer/dataset/traffic/ \
 --data_path traffic.csv \
 --task_id traffic \
 --model $model \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 862 \
 --des 'Exp' \
 --itr 1 \
 --train_epochs 3

done

done