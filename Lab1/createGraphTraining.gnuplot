set term png
set datafile separator ","
set output "plot_training.png"
set xlabel "Training set size"
set ylabel "% accuracy"
plot "perceptron1vr_train.csv" using 1:2 title "Training accuracy" with lines, \
 "perceptron1vr_test.csv" using 1:2 title "Test accuracy" with lines
