
straight="/afs/inf.ed.ac.uk/group/project/dnn_tts/lstm/dnn_tts_public/tools/WORLD/build"
sptk="/afs/inf.ed.ac.uk/group/project/dnn_tts/tools/dnn_tts/tools/SPTK-3.7/bin"

wav_dir="/afs/inf.ed.ac.uk/group/project/dnn_tts/lstm/dnn_tts_public/tools/WORLD/wav_test"


for file in $wav_dir/*.wav #.wav
do
    filename="${file##*/}"
    file_id="${filename%.*}"
   
    echo $file_id
    
    
    $straight/analysis $wav_dir/$file_id.wav $wav_dir/$file_id.f0 $wav_dir/$file_id.sp $wav_dir/$file_id.ap
    
    $sptk/x2x +df $wav_dir/$file_id.sp | $sptk/sopr -R -m 32768.0 | $sptk/mcep -a 0.77 -m 59 -l 2048 -e 1.0E-8 -j 0 -f 0.0 -q 3 > $wav_dir/$file_id.mgc

    $sptk/mgc2sp -a 0.77 -g 0 -m 59 -l 2048 -o 2 $wav_dir/$file_id.mgc | $sptk/sopr -d 32768.0 -P | $sptk/x2x +fd > $wav_dir/$file_id.resyn.sp
    
    $straight/synth 2048 44100 $wav_dir/$file_id.f0 $wav_dir/$file_id.resyn.sp $wav_dir/$file_id.ap $wav_dir/$file_id.resyn.wav
done

