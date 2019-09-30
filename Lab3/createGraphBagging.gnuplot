set term png
set datafile separator ","
set output "plot_bagging.png"
set xlabel "Number of Classifiers"
set ylabel "% Accuracy"
plot "bagging_train.csv" using 1:2 title "train" with lines, \
	"bagging_val.csv" using 1:2 title "validation" with lines, \
	"bagging_test.csv" using 1:2 title "test" with lines,	
