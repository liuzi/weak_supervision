clamp_for_n2c2_folder='n2c2'
rm -r $clamp_for_n2c2_folder
mkdir $clamp_for_n2c2_folder


clamp_raw_result_path="/data/liu/mimic3/CLAMP_NER/ner-attribute/output/discharge_summary"
prefix="../N2C2/all_txt/"
suffix=".txt"

for file in "../N2C2/all_txt"/*
do
    file_base=${file#$prefix}
    file_base=${file_base%$suffix}
    files=`find $clamp_raw_result_path/ -type f -name "*_$file_base*.txt" | sort -nr | head -1`
    cp ${files[0]} $clamp_for_n2c2_folder/$file_base.txt
    # done
    # echo $file_base
done
    # for i in 'grep -l * src_dir/201701*`