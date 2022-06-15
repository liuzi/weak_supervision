hoscourse_for_n2c2_folder='n2c2_hoscourse'
rm -r $hoscourse_for_n2c2_folder
mkdir $hoscourse_for_n2c2_folder


hoscourse_raw_result_path="/data/liu/mimic3/CLAMP_NER/input/*/HOSPITAL_COURSE/"
prefix="../N2C2/all_txt/"
suffix=".txt"

for file in "../N2C2/all_txt"/*
do
    file_base=${file#$prefix}
    file_base=${file_base%$suffix}
    files=`find $hoscourse_raw_result_path/ -type f -name "*_$file_base*.txt" | sort -nr | head -1`
    cp ${files[0]} $hoscourse_for_n2c2_folder/$file_base.txt
    # done
    # echo $file_base
done
    # for i in 'grep -l * src_dir/201701*`

## Not found txt, add manually
# < 100039.txt
# 50d48
# < 102527.txt
# 142d139
# < 106945.txt
# 178d174
# < 109330.txt
# 222d217
# < 111840.txt
# 235,236d229
# < 112445.txt
# < 112446.txt
# 262d254
# < 114211.txt
# 283d274
# < 115199.txt
# 288d278
# < 115391.txt
# 298d287
# < 116011.txt
# 317d305
# < 118496.txt
# 319d306
# < 118564.txt
# 346d332
# < 123204.txt
# 389d374
# < 139825.txt
# 407d391
# < 149614.txt
# 427d410
# < 157103.txt
# 444d426
# < 166834.txt