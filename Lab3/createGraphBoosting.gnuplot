set term png
set datafile separator ","
set output "plot_boosting.png"
set xlabel "Number of Classifiers"
set ylabel "% Accuracy"
plot "boosting_train.csv" using 1:2 title "train" with lines, \
	"boosting_val.csv" using 1:2 title "validation" with lines, \
	"boosting_test.csv" using 1:2 title "test" with lines,