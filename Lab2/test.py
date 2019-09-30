import sys
from tasks import *
import numpy as np
# Script Usage: python3 test.py <task_num> <seed>
# Read task number and seed value from command line

task={
	1 : taskSquare,
	2 : taskSemiCircle,
	3 : taskMnist,
	4 : taskCifar10,
}
if __name__ == "__main__":
	task_num=int(sys.argv[1])
	seed=int(sys.argv[2])

	np.random.seed(int(seed))
	if(task_num <= 2):
		task[task_num](True)
	else:
		task[task_num]()
