set term png
set datafile separator ","
set output "plot_iterations.png"
set xlabel "Number of data-points seen"
set ylabel "% accuracy"
plot "perceptron1Iterations.csv" using 1:2 title "Test Accuracy" with lines, \
 "perceptron1IterationsTrain.csv" using 1:2 title "Train Accuracy" with lines

