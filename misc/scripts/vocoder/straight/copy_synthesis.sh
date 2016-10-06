#!/bin/sh

# top merlin directory
merlin_dir="/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/test/merlin"

# tools directory
straight="${merlin_dir}/tools/bin/straight"
sptk="${merlin_dir}/tools/bin/SPTK-3.9"

# input audio directory
wav_dir="${merlin_dir}/tools/WORLD/wav_test"

# Output features directory
out_dir="${merlin_dir}/tools/WORLD/wav_test"

raw_dir="${out_dir}/raw"
sp_dir="${out_dir}/sp"
mgc_dir="${out_dir}/mgc"
ap_dir="${out_dir}/ap"
bap_dir="${out_dir}/bap"
f0_dir="${out_dir}/f0"
lf0_dir="${out_dir}/lf0"
resyn_dir="${out_dir}/resyn_dir"

mkdir -p $out_dir
mkdir -p $raw_dir
mkdir -p $sp_dir
mkdir -p $mgc_dir
mkdir -p $ap_dir
mkdir -p $bap_dir
mkdir -p $f0_dir
mkdir -p $lf0_dir
mkdir -p $resyn_dir

# initializations
fs=16000

if [ "$fs" -eq 16000 ]
then
nFFT=1024
alpha=0.58
fi

if [ "$fs" -eq 48000 ]
then
nFFT=4096
alpha=0.77
fi

mcsize=59
order=24
nFFTHalf=$((1 + $nFFT / 2)) 
fshift=5


for file in $wav_dir/*.wav #.wav
do
    filename="${file##*/}"
    file_id="${filename%.*}"
   
    echo $file_id

    sox $wav_dir/$file_id.wav -b 16 -c 1 -r $fs -t raw $raw_dir/$file_id.raw

    ### STRAIGHT ANALYSIS -- extract vocoder parameters ###

    ### extract f0, sp, ap ### 
    $straight/tempo -nmsg -maxf0 400 -uf0 400 -minf0 50 -lf0 50 -f0shift $fshift -f $fs -raw $raw_dir/$file_id.raw ${f0_dir}/$file_id.f0
    $straight/straight_bndap -nmsg -f $fs -fftl $nFFT -apord $nFFTHalf -shift $fshift -f0shift $fshift -float -f0file $f0_dir/$file_id.f0 -raw $raw_dir/$file_id.raw $ap_dir/$file_id.ap
    $straight/straight_mcep -nmsg -f $fs -fftl $nFFT -apord $nFFTHalf -shift $fshift -f0shift $fshift -order $mcsize -f0file ${f0_dir}/$file_id.f0 -pow -float -raw $raw_dir/$file_id.raw $sp_dir/$file_id.sp
    
    ### convert f0 to lf0 ###
    $sptk/x2x +af ${f0_dir}/$file_id.f0 | $sptk/sopr -magic 0.0 -LN -MAGIC -1.0E+10 > ${lf0_dir}/$file_id.lf0
   
    ### convert sp to mgc ###
    $sptk/mcep -a $alpha -m $mcsize -l $nFFT -e 1.0E-8 -j 0 -f 0.0 -q 3 $sp_dir/$file_id.sp > $mgc_dir/$file_id.mgc
    
    ### convert ap to bap ###
    $sptk/mcep -a $alpha -m $order -l $nFFT -e 1.0E-8 -j 0 -f 0.0 -q 1 $ap_dir/$file_id.ap > $bap_dir/$file_id.bap
    
    ## 2nd version of extracting bap -- not recommended
    #$straight/straight_bndap -nmsg -f $fs -fftl $nFFT -apord $nFFTHalf -shift $fshift -f0shift $fshift -bndap -float -f0file $f0_dir/$file_id.f0 -raw $raw_dir/$file_id.raw $bap_dir/$file_id.bap

    ### STRAIGHT Re-synthesis -- reconstruction of parameters ###

    ### convert lf0 to f0 ###
    $sptk/sopr -magic -1.0E+10 -EXP -MAGIC 0.0 $lf0_dir/$file_id.lf0 | $sptk/x2x +fa > $resyn_dir/$file_id.f0
   
    ### convert mgc to sp ###
    $sptk/mgc2sp -a $alpha -g 0 -m $mcsize -l $nFFT -o 2 $mgc_dir/$file_id.mgc > $resyn_dir/$file_id.sp

    ### convert bap to ap ###
    $sptk/mgc2sp -a $alpha -g 0 -m $order -l $nFFT -o 0 $bap_dir/$file_id.bap > $resyn_dir/$file_id.ap
    
    ## if using v2 of bap
    #$straight/bndap2ap $bap_dir/$file_id.bap > $resyn_dir/$file_id.ap

    $straight/synthesis_fft -f $fs -spec -fftl $nFFT -shift $fshift -sigp 1.2 -cornf 4000 -float -apfile $resyn_dir/$file_id.ap $resyn_dir/$file_id.f0 $resyn_dir/$file_id.sp $wav_dir/$file_id.resyn.wav
   
done

rm -rf $sp_dir $mgc_dir $f0_dir $lf0_dir $bap_dir $ap_dir
rm -rf $resyn_dir
