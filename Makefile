tr-vad:
	python vad_analyzer.py --input_dir /Users/javad/Projects/qwise/datasets/noise/NOIZEUS/rand_noise \
	--output_dir ./test --enable_silero --enable_trvad \
	--trvad_ckpt_path ./Tr-VAD/checkpoint/weights_10_acc_97.09.pth

sinc-vad:
	python vad_analyzer.py --input_dir datasets/noise/NOIZEUS/resampled_16k \
	--output_dir ./test --enable_silero --enable_sincqdr --output_dir ./test

clean:
	rm -r test/

demo:
	python demo.py datasets/noise/NOIZEUS/8khz/sp01_airport_sn0.wav  -o ./test/clean.wav --plot-save ./test/result.png
