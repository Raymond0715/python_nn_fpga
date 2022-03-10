#cat ckpt_dat/yolo/bias_208_16_shift_process_16bit.dat >> ckpt_dat/on_board/bias_weight_l345.bin
#cat ckpt_dat/yolo/bias_208_16_shift_process_16bit.dat >> ckpt_dat/on_board/bias_weight_l345.bin
#cat ckpt_dat/yolo/bias_104_32_shift_process.dat >> ckpt_dat/on_board/bias_weight_l345.bin
#cat ckpt_dat/yolo/weight_208_16_shift_process_16bit_64paral.dat >> ckpt_dat/on_board/bias_weight_l345.bin
#cat ckpt_dat/yolo/weight_104_32_shift_process.dat >> ckpt_dat/on_board/bias_weight_l345.bin

cat ckpt_dat/yolo/bias_208_16_shift_process_sim.txt           >> ckpt_dat/sim/bias_weight_l345_sim.txt
cat ckpt_dat/yolo/bias_208_16_shift_process_sim.txt           >> ckpt_dat/sim/bias_weight_l345_sim.txt
cat ckpt_dat/yolo/bias_104_32_shift_process_sim.txt           >> ckpt_dat/sim/bias_weight_l345_sim.txt
cat ckpt_dat/yolo/weight_208_16_shift_process_64paral_sim.txt >> ckpt_dat/sim/bias_weight_l345_sim.txt
cat ckpt_dat/yolo/weight_104_32_shift_process_sim.txt         >> ckpt_dat/sim/bias_weight_l345_sim.txt
