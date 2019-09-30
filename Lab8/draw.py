from time import time
from tkinter   import *
from random import randint

screenSize = 500
################################ DRAWING METHODS #################################

def randColor():
	return "#%06x" % randint(0,0xFFFFFF)

def drawTour(cities, path):
	addToCanvas(cities, path)
	canvas.update()
	root.mainloop()

def addToCanvas(cities, path):
	min_x = cities[path[0]].x
	min_y = cities[path[0]].y
	max_x = cities[path[0]].x
	max_y = cities[path[0]].y	
	for i in range(1, len(path)):
		if cities[path[i]].x < min_x:
			min_x = cities[path[i]].x
		if cities[path[i]].y < min_y:
			min_y = cities[path[i]].y

		if cities[path[i]].x > max_x:
			max_x = cities[path[i]].x
		if cities[path[i]].y > min_y:
			max_y = cities[path[i]].y


	for i in range( len( path ) ):
		c = cities[path[i-1]]
		c_next = cities[path[i]]

		scaled_x = (c.x - min_x) / (max_x - min_x) * screen_width/2 + 20
		scaled_y = (c.y - min_y) / (max_y - min_y) * screen_height/2 + 20
		scaled_x_next = (c_next.x - min_x) / (max_x - min_x) * screen_width/2 + 20
		scaled_y_next = (c_next.y - min_y) / (max_y - min_y) * screen_height/2 + 20
		print(scaled_x, scaled_y, scaled_x_next, scaled_y_next)
		canvas.create_oval( scaled_x - 4 , scaled_y - 4 , scaled_x + 4  , scaled_y + 4 , fill = randColor() , outline = 'black' )
		canvas.create_oval( scaled_x_next - 4 , scaled_y_next - 4 , scaled_x_next + 4  , scaled_y_next + 4 , fill = randColor() , outline = 'black' )

		canvas.create_text(scaled_x+8, scaled_y+8, font=("Helvetica", 10), text=str(path[i]))
		canvas.create_line( scaled_x, scaled_y , scaled_x_next, scaled_y_next , fill = 'black' )


root = Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
canvas = Canvas( root , width = screen_width-100, height = screen_height-100 , bg = 'white' )
canvas.pack()

# add scroolbar and link to canvas
scrollbar=Scrollbar(root)
scrollbar.pack(side=RIGHT,fill=Y)
canvas.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=canvas.yview)

if __name__ == "__main__":
	main()
