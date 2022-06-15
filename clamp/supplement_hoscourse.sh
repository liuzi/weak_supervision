# diff <(cd n2c2_clamp; ls) <(cd n2c2_hoscourse; ls) > files
# echo $files

# for file in (diff <(cd n2c2_clamp; ls) <(cd n2c2_hoscourse; ls))
# for file in (
# files=find "$Dir1/" "$Dir2/" "$Dir2/" -printf '%P\n' | sort | uniq -u
# echo $files
# do
    # echo $file
# done

keyword="hospital course"
shopt -s nocasematch

for f1 in n2c2_clamp/*; do
    f2="n2c2_hoscourse/$( basename "$f1" )"
    if [ ! -e "$f2" ]; then
        missed_f="../N2C2/all_txt/$( basename "$f1" )"
        egrep -n "^[A-Z][A-Za-z \t]+:$" $missed_f| while read -r value ; do
        # egrep -n "^\u\S*( \u\S*)*( \S*)?( \u\S*)*( \S*)?( \u\S*)*$" $missed_f| while read -r line ; do
            # num, txt = line
            # echo $var
            # echo $value
            key = $value | cut -d';' -f1
            # arrvalue=(${value//:/ })
            # echo ${arrvalue[0]}
            # echo ${arrvalue[1]}
            # if $line =~ $keyword; then
                # echo $line
            # fi
        done
        # echo $missed_f
    fi
done