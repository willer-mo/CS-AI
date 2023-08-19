from turtle import *
import colorsys
bgcolor("black")
tracer(15)
pensize(5)
h = 0
for i in range(330):
    c = colorsys.hsv_to_rgb(h, 1, 1)
    h += 0.05
    pencolor(c)
    fillcolor("black")
    begin_fill()
    for j in range(4):
        fd(i * 1.2)
        fd(i * 1.2)
        rt(40)
        fd(200)
        rt(500)
    rt(333)
    end_fill()
done()
