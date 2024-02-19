lim=100
while :
do
    sleep 10
    a=`nvidia-smi --query-gpu=memory.used --format=csv|cut -f 1 -d ' ' | tail -n 8`
    g0=`echo $a |cut -f 1 -d ' '`
    g1=`echo $a |cut -f 2 -d ' '`
    g2=`echo $a |cut -f 3 -d ' '`
    g3=`echo $a |cut -f 4 -d ' '`
    g4=`echo $a |cut -f 5 -d ' '`
    g5=`echo $a |cut -f 6 -d ' '`
    g6=`echo $a |cut -f 7 -d ' '`
    g7=`echo $a |cut -f 8 -d ' '`
    if [ $g0 -lt $lim ] && [ $g1 -lt $lim ] && [ $g2 -lt $lim ] && [ $g3 -lt $lim ] &&  [ $g4 -lt $lim ] && [ $g5 -lt $lim ] && [ $g6 -lt $lim ] && [ $g7 -lt $lim ]; then
        bash gpu_boom.sh
        exit
    fi
done
