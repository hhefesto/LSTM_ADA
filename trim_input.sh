declare -a arr=("ADABTC" "ETCBTC" "IOTABTC" "NANOBTC" "TRXBTC" "XRPBTC" "BNBBTC" "DASHBTC" "ETHBTC" "LSKBTC" "NEOBTC" "XLMBTC" "BCCBTC" "BTCUSDT" "EOSBTC" "ICXBTC" "LTCBTC" "QTUMBTC" "XMRBTC")

for i in "${arr[@]}"
do
    echo "$i"
    FILE="$i.csv"
    echo $FILE
    TRIMED="_trimed.csv"
    FILE_T="$i$TRIMED"
    echo $FILE_T
    tail -n 100000 $FILE >  $FILE_T
    echo ""
    wc -l "$i_trimed.csv"
    echo ""
    # or do whatever with individual element of the array
done
